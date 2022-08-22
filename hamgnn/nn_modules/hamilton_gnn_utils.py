from pathlib import Path
import pkgutil
import importlib
from inspect import isclass
from abc import abstractmethod
import itertools
import numpy

import torch
import torchmetrics
import torch.nn.functional as F
import torch_geometric as torch_g
import torch_scatter
import pytorch_lightning as torch_lightning

from hamgnn.HamiltonSolver import DataUtils, HamiltonSolver
import hamgnn.Evaluation as Evaluation


def list_of_gnn_model_classes():
    model_classes = []
    for (_, module_name, _) in pkgutil.iter_modules([Path(__file__).resolve().parent]):
        module = importlib.import_module(f"{__name__}.{module_name}")
        model_classes.extend([var for var_name, var in vars(module).items() if isclass(var) and issubclass(var, HamFinderGNN)])
    return model_classes


class WalkUpdater:
    @staticmethod
    def batch_prepare_for_first_step(d: torch_g.data.Batch, starting_nodes):
        d.x = torch.zeros([d.num_nodes, 3], device=d.edge_index.device)
        if starting_nodes is not None:
            d.x[starting_nodes, ...] = 1
        return d

    @staticmethod
    def update_batch_states(d: torch_g.data.Data, current_nodes):
        previous = torch.eq(d.x[..., 1], 1)
        if torch.any(current_nodes < 0).item() or torch.any(current_nodes > d.num_nodes).item():
            raise Exception("Illegal choice of next step node when updating graph mask!")
        d.x[previous, 1] = 0
        d.x[current_nodes, 1] = 1
        d.x[current_nodes, 2] = 1


class HamFinderGNN(HamiltonSolver, torch_lightning.LightningModule):
    def __init__(self, graph_updater_class: WalkUpdater,
                 inference_batch_size=8, starting_learning_rate=1e-4, optimizer_class=None, optimizer_hyperparams=None, lr_scheduler_class=None, lr_scheduler_hyperparams=None, val_dataloader_tags=None):
        super(HamFinderGNN, self).__init__()
        self.graph_updater = graph_updater_class()
        self._inference_batch_size = inference_batch_size
        self.optimizer_class = optimizer_class
        self.optimizer_hyperparams = optimizer_hyperparams
        self.learning_rate = starting_learning_rate
        self.lr_scheduler_class = lr_scheduler_class
        self.lr_scheduler_hyperparams = lr_scheduler_hyperparams
        self.val_dataloader_tags = val_dataloader_tags

        self.log_train_tag = "train"
        self.log_val_tag = "val"
        self.log_test_tag = "test"
        self.accuracy_metrics_central_dict = dict()

    @abstractmethod
    def raw_next_step_logits(self, d: torch_g.data.Batch) -> torch.Tensor:
        pass

    def next_step_logits_masked_over_neighbors(self, d: torch_g.data.Batch) -> torch.Tensor:
        logits = self.raw_next_step_logits(d)
        if torch.all(torch.lt(d.x[..., 1], 0.5)):
            return logits
        else:
            return DataUtils._mask_neighbor_logits(logits, d)

    def scattered_logits_to_probabilities(self, graph_batch: torch_g.data.Batch, logits=None):
        return torch_scatter.scatter_softmax(logits, graph_batch.batch)

    def prepare_for_first_step(self, d: torch_g.data.Batch, start_batch):
        return self.graph_updater.batch_prepare_for_first_step(d, start_batch)

    def update_state(self, d: torch_g.data.Data, current_batch):
        return self.graph_updater.update_batch_states(d, current_batch)

    @abstractmethod
    def init_graph(self, d) -> None:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    def _neighbor_prob_and_greedy_choice_for_batch(self, d: torch_g.data.Batch):
        logits = self.next_step_logits_masked_over_neighbors(d)
        p = self.scattered_logits_to_probabilities(d, logits)
        graph_sizes = [g.num_nodes for g in d.to_data_list()]
        max_size = max(graph_sizes)
        p = torch.stack([F.pad(x, (0, max_size - x.shape[0]), "constant", -1) for x in torch.split(p, graph_sizes)])
        choice = torch.argmax(
            torch.isclose(p, torch.max(p, dim=-1)[0][..., None])
            * (p + torch.randperm(p.shape[-1], device=p.device)[None, ...]), dim=-1)
        choice += torch.tensor(list(itertools.accumulate(graph_sizes[:-1], initial=0)))

        return p, choice

    class _RunInstructions:
        def on_algorithm_start_fn(self, batch_graph):
            start_step_number = 0
            return start_step_number

        def choose_next_step_fn(self, batch_graph, current_nodes, step_number, is_algorithm_stopped_mask):
            pass

        def update_algorithm_stopped_mask(self, all_previous_choices, current_choice, is_algorithm_stopped_mask, step_number):
            if len(all_previous_choices) > 0:
                already_visited_mask = torch.any(
                    torch.stack([torch.eq(x, current_choice) for x in all_previous_choices], -1), -1)
                is_algorithm_stopped_mask = torch.maximum(is_algorithm_stopped_mask, already_visited_mask)
            return is_algorithm_stopped_mask

    def _run_on_graph_batch(self, batch_graph: torch_g.data.Batch, run_instructions: _RunInstructions):
        nodes_per_graph = [d.num_nodes for d in batch_graph.to_data_list()]
        max_steps_in_a_cycle = max(nodes_per_graph) + 1
        all_choices = []
        next_step_choices = None
        current_nodes = None
        is_algorithm_stopped_mask = torch.zeros(
            [batch_graph.num_graphs], device=batch_graph.edge_index.device, dtype=torch.bool)

        start_step_number = run_instructions.on_algorithm_start_fn(batch_graph)

        for step_number in range(start_step_number, max_steps_in_a_cycle):
            if step_number in [0, 1]:
                self.prepare_for_first_step(batch_graph, current_nodes)
            else:
                self.update_state(batch_graph, current_nodes[current_nodes != -1])

            next_step_choices = run_instructions.choose_next_step_fn(batch_graph, current_nodes, step_number, is_algorithm_stopped_mask)
            next_step_choices = torch.logical_not(is_algorithm_stopped_mask) * next_step_choices - is_algorithm_stopped_mask * torch.ones_like(next_step_choices)
            current_nodes = next_step_choices

            is_algorithm_stopped_mask = run_instructions.update_algorithm_stopped_mask(all_choices, current_nodes, is_algorithm_stopped_mask, step_number)
            all_choices.append(next_step_choices)

            if torch.all(is_algorithm_stopped_mask).item():
                break
        walks = torch.stack(all_choices, -1)
        if walks.shape[-1] != max_steps_in_a_cycle:
            walks = F.pad(walks, (0, max_steps_in_a_cycle - walks.shape[-1]), value=-1)
        return walks

    def batch_run_greedy_neighbor(self, batch_graph: torch_g.data.Batch):
        class GreedyRunInstructions(self._RunInstructions):
            def __init__(self, gnn_model) -> None:
                self.gnn_model = gnn_model

            def on_algorithm_start_fn(self, batch_graph):
                self.gnn_model.init_graph(batch_graph)
                start_index = 0
                return start_index

            def choose_next_step_fn(self, batch_graph, current_nodes, step_number, is_algorithm_stopped_mask):
                _, next_step_choices = self.gnn_model._neighbor_prob_and_greedy_choice_for_batch(batch_graph)
                return next_step_choices

        walks = self._run_on_graph_batch(batch_graph, GreedyRunInstructions(self))
        return walks

    @staticmethod
    def _convert_batch_walk_tensor_into_solution_list(batch_walks_tensor, graphs_shift_inside_batch):
        batch_walks_tensor -= graphs_shift_inside_batch[:, None]
        batch_walks_tensor[batch_walks_tensor < 0] = -1
        raw_walks = [[int(node.item()) for node in walk] for walk in batch_walks_tensor]
        walks = []
        for walk in raw_walks:
            walk_end_index = walk.index(-1) if -1 in walk else len(walk)
            walks.append(walk[:walk_end_index])
        return walks

    @staticmethod
    def _unpack_graph_batch_dict(graph_batch_dict) -> tuple[torch_g.data.Batch, list[list[int]]]:
        list_of_graph_edge_indexes = graph_batch_dict["list_of_edge_indexes"]
        list_of_graph_num_nodes = graph_batch_dict["list_of_graph_num_nodes"]
        teacher_paths = graph_batch_dict.get("teacher_paths")
        graphs = []
        for edge_index, num_nodes in zip(list_of_graph_edge_indexes, list_of_graph_num_nodes):
            graph = torch_g.data.Data(edge_index=edge_index)
            graph.num_nodes = num_nodes
            graphs.append(graph)
        batch_graph = torch_g.data.Batch.from_data_list(graphs)
        return batch_graph, teacher_paths

    def solve_batch_graph(self, batch_graph, subgraph_sizes=None):
        if subgraph_sizes is None:
            subgraph_sizes = [g.num_nodes for g in batch_graph.to_data_list()]

        batch_shift = numpy.cumsum([0] + [subgraph_size for subgraph_size in subgraph_sizes[:-1]])
        walks_tensor = self.batch_run_greedy_neighbor(batch_graph)
        return self._convert_batch_walk_tensor_into_solution_list(walks_tensor, batch_shift)

    def get_batch_size_for_multi_solving(self):
        return 1

    def solve_graphs(self, graphs):
        graph_iterator = iter(graphs)
        batch_size = self.get_batch_size_for_multi_solving()
        batch_generator = ([first] + [d for d in itertools.islice(graph_iterator, batch_size - 1)]
                           for first in graph_iterator)

        walks = []
        for list_of_graphs in batch_generator:
            subgraph_sizes = [g.num_nodes for g in list_of_graphs]
            batch_graph = torch_g.data.Batch.from_data_list(list_of_graphs)
            walks.extend(self.solve_batch_graph(batch_graph, subgraph_sizes))
        for walk in walks:
            if len(walk) > 1 and walk[-1] == walk[-2]:
                walk.pop(-1)
        return walks

    def update_accuracy_metrics(self, accuracy_metrics_tag: str, graphs: list[torch_g.data.Data], solutions: list[list[int]]):
        if accuracy_metrics_tag not in self.accuracy_metrics_central_dict:
            self.accuracy_metrics_central_dict[accuracy_metrics_tag] = torch.nn.ModuleDict({
                name: torchmetrics.Accuracy(compute_on_step=False)
                for name in ("hamiltonian_cycle", "hamiltonian_path", "90_perc_cycle", "90_perc_path")})
        accuracy_metrics = self.accuracy_metrics_central_dict[accuracy_metrics_tag]

        evals = Evaluation.EvaluationScores.evaluate(graphs, solutions)
        df_scores = Evaluation.EvaluationScores.compute_scores(evals)
        for metric_name, column_name in [("hamiltonian_cycle", "is_ham_cycle"), ("hamiltonian_path", "is_ham_path"),
                                         ("90_perc_cycle", "is_approx_ham_cycle"), ("90_perc_path", "is_approx_ham_path")]:
            if metric_name in accuracy_metrics:
                metric = accuracy_metrics[metric_name]
                prediction = torch.tensor(df_scores[column_name], dtype=torch.float)
                metric.update(prediction, torch.ones_like(prediction, dtype=torch.int))

    def log_accuracy_metric(self, accuracy_metrics_tag):
        accuracy_metrics = self.accuracy_metrics_central_dict[accuracy_metrics_tag]
        for metric_name, metric in accuracy_metrics.items():
            self.log(f"{accuracy_metrics_tag}/{metric_name}", metric.compute())
            metric.reset()

    def get_validation_dataloader_tag(self, dataloader_idx):
        dataloader_subtag = f"/{dataloader_idx}"
        if self.val_dataloader_tags is None:
            if dataloader_idx == 0:
                dataloader_subtag = ""
        else:
            if dataloader_idx < len(self.val_dataloader_tags):
                dataloader_subtag = f"/{self.val_dataloader_tags[dataloader_idx]}"
        return f"{self.log_val_tag}{dataloader_subtag}"

    def on_validation_epoch_end(self) -> None:
        for tag, accuracy_metrics in self.accuracy_metrics_central_dict.items():
            if tag.startswith(self.log_val_tag):
                self.log_accuracy_metric(tag)

    def _compute_and_update_accuracy_metrics(self, graph_batch_dict, accuracy_metrics_tag):
        batch_graph, _ = self._unpack_graph_batch_dict(graph_batch_dict)
        walks = self.solve_batch_graph(batch_graph)
        graph_list = batch_graph.to_data_list()
        self.update_accuracy_metrics(accuracy_metrics_tag, graph_list, walks)

    def test_step(self, graph_batch_dict, batch_idx, dataloader_idx=0):
        with torch.no_grad():
            self._compute_and_update_accuracy_metrics(graph_batch_dict, self.log_test_tag)

    def on_test_epoch_end(self, *args, **kwargs) -> None:
        self.log_accuracy_metric(self, self.log_test_tag)
        return super().on_test_epoch_end()

    def configure_optimizers(self):
        if self.optimizer_class is None:
            optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)
        else:
            optimizer = self.optimizer_class(self.parameters(), **self.optimizer_hyperparams)
        config_dict = {"optimizer": optimizer}

        if self.lr_scheduler_class is not None:
            lr_scheduler = self.lr_scheduler_class(optimizer, **self.lr_scheduler_hyperparams)
            config_dict.update({"lr_scheduler": lr_scheduler})

        return config_dict


class HamCycleFinderWithValueFunction(HamFinderGNN):
    @abstractmethod
    def raw_next_step_logits_and_value_function(self, d:torch_g.data.Data) -> tuple[torch.Tensor, torch.Tensor]:
        pass