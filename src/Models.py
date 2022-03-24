import itertools
import os.path
from abc import ABC, abstractmethod
import numpy

import torch
import torch.nn.functional as F
import torch_scatter
import torch_geometric as torch_g
import torchinfo
import pytorch_lightning as torch_lightning
import torchmetrics

from src.HamiltonSolver import DataUtils, HamiltonSolver
from src.NN_modules import ResidualMultilayerMPNN, MultilayerGatedGCN
from src.data.GraphDataset import BatchedSimulationStates, GraphBatchExample, GraphExample, SimulationState, get_shifts_for_graphs_in_batch
import src.Evaluation as Evaluation
import src.solution_scorers as scorers


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

        def update_algorithm_stopped_mask(self, all_previous_choices, current_choice, is_algorithm_stopped_mask):
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

            is_algorithm_stopped_mask = run_instructions.update_algorithm_stopped_mask(all_choices, current_nodes, is_algorithm_stopped_mask)
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
        dataloader_subtag = f"{dataloader_idx}"
        if self.val_dataloader_tags is not None and dataloader_idx < len(self.val_dataloader_tags):
            dataloader_subtag = self.val_dataloader_tags[dataloader_idx]
        return f"{self.log_val_tag}/{dataloader_subtag}"

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
            optimizer = self.optimizer_class(**self.optimizer_hyperparams)
        config_dict = {"optimizer": optimizer}

        if self.lr_scheduler_class is not None:
            lr_scheduler = self.lr_scheduler_class(optimizer, **self.lr_scheduler_hyperparams)
            config_dict.update({"lr_scheduler": lr_scheduler})

        return config_dict


class HamCycleFinderWithValueFunction(HamFinderGNN):
    @abstractmethod
    def raw_next_step_logits_and_value_function(self, d:torch_g.data.Data) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class EncodeProcessDecodeAlgorithm(HamFinderGNN):
    def _construct_processor(self):
        return ResidualMultilayerMPNN(dim=self.hidden_dim, message_dim=self.hidden_dim, edges_dim=1,
                                      nr_layers=self.processor_depth)

    def _construct_encoder_and_decoder(self):
        encoder_nn = torch.nn.Sequential(torch.nn.Linear(3 + self.hidden_dim, self.hidden_dim))
        decoder_nn = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim + self.hidden_dim, self.out_dim))
        return encoder_nn, decoder_nn

    def __init__(self, processor_depth=3, in_dim=1, out_dim=1, hidden_dim=32, graph_updater_class=WalkUpdater,
                 loss_type="mse", **kwargs):
        super().__init__(graph_updater_class, **kwargs)
        self.save_hyperparameters()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.processor_depth = processor_depth
        assert loss_type in ["mse", "entropy"]
        self.loss_type = loss_type

        self.encoder_nn, self.decoder_nn = self._construct_encoder_and_decoder()
        self.processor_nn = self._construct_processor()
        self.initial_h = torch.nn.Parameter(torch.rand(self.hidden_dim))

    def description(self):
        return f"encoder: {torchinfo.summary(self.encoder_nn, verbose=0, depth=5)}\n" \
               f"processor (hidden dim={self.hidden_dim}):" \
               f" {torchinfo.summary(self.processor_nn, verbose=0, depth=5)}\n" \
               f"decoder: {torchinfo.summary(self.decoder_nn, verbose=0, depth=5)}"

    def raw_next_step_logits(self, d: torch_g.data.Data):
        d.z = self.encoder_nn(torch.cat([d.x, d.h], dim=-1))
        d.h = self.processor_nn(d.z, d.edge_index, d.edge_attr)
        return torch.squeeze(self.decoder_nn(torch.cat([d.z, d.h], dim=-1)), dim=-1)

    def init_graph(self, d):
        #TODO edge features are useless here but are needed because of how layers are implemented at the moment
        d.edge_attr = torch.ones_like(d.edge_index[0, ...].unsqueeze(-1), dtype=self.initial_h.dtype)
        d.h = torch.stack([self.initial_h for _ in range(d.num_nodes)], dim=-2)

    def forward(self, d: torch_g.data.Data):
        return self.next_step_logits_masked_over_neighbors(d)

    def training_step(self, graph_batch_dict):
        batch_graph, teacher_paths = self._unpack_graph_batch_dict(graph_batch_dict)

        class TrainingRunInstructions(self._RunInstructions):
            def __init__(self, gnn_model: HamFinderGNN, graph_batch_dict) -> None:
                self.gnn_model: HamFinderGNN = gnn_model
                self.batch_graph, teacher_paths = self.gnn_model._unpack_graph_batch_dict(graph_batch_dict)
                self.loss = torch.zeros(1, device=self.batch_graph.edge_index.device)
                graph_shifts = get_shifts_for_graphs_in_batch(batch_graph)
                self.teacher_tensor = torch.stack(teacher_paths, 0)
                self._logits_per_step = []

                if self.gnn_model.loss_type == "mse":
                    mse_weights = F.normalize(F.one_hot(batch_graph.batch, batch_graph.num_graphs).float(), 1, 0).sum(-1)
                    def _compute_loss(logits, probabilities, step_number, is_algorithm_stopped_mask):
                        self._logits_per_step.append(logits)
                        mse_loss = torch.nn.MSELoss(reduction="none")
                        teacher_p = torch.zeros_like(probabilities)
                        teacher_p[self.teacher_tensor[:, step_number]] = 1
                        return (mse_loss(probabilities, teacher_p) * mse_weights).sum()
                elif self.gnn_model.loss_type == "entropy":
                    def _compute_loss(logits, probabilites, step_number, is_algorithm_stopped_mask):
                        self._logits_per_step.append(logits)
                        entropy_loss = torch.nn.CrossEntropyLoss()
                        subgraph_losses = []
                        for subgraph_index in range(len(graph_shifts)):
                            graph_start_index = graph_shifts[subgraph_index]
                            graph_end_index = graph_shifts[subgraph_index+1] if subgraph_index + 1 < len(graph_shifts) else self.batch_graph.num_nodes
                            _graph_logits = logits[graph_start_index: graph_end_index].unsqueeze(0)
                            subgraph_losses.append(
                                entropy_loss(_graph_logits, self.teacher_tensor[subgraph_index: subgraph_index + 1, step_number] - graph_start_index))
                        return torch.stack(subgraph_losses).sum()
                self.compute_loss = _compute_loss

            def on_algorithm_start_fn(self, batch_graph):
                self.gnn_model.init_graph(batch_graph)
                self.gnn_model.prepare_for_first_step(batch_graph, None)
                _ = self.gnn_model.raw_next_step_logits(batch_graph)
                self.gnn_model.graph_updater.batch_prepare_for_first_step(batch_graph, self.teacher_tensor[:, 0])
                start_index = 1
                return start_index

            def choose_next_step_fn(self, batch_graph, current_nodes, step_number, is_algorithm_stopped_mask):
                logits = self.gnn_model.raw_next_step_logits(batch_graph)
                probabilites = self.gnn_model.scattered_logits_to_probabilities(batch_graph, logits)
                self.loss += self.compute_loss(logits, probabilites, step_number, is_algorithm_stopped_mask)
                return self.teacher_tensor[:, step_number]

            def update_algorithm_stopped_mask(self, all_previous_choices, current_choice, is_algorithm_stopped_mask):
                return is_algorithm_stopped_mask # No need for update, because of teacher forcing all batch algortithms will run util the end.

        run_instructions = TrainingRunInstructions(self, graph_batch_dict)
        self._run_on_graph_batch(batch_graph, run_instructions)

        avg_loss = run_instructions.loss / run_instructions.teacher_tensor.nelement()
        self.logits_per_step = run_instructions._logits_per_step
        self.log("train/loss", avg_loss)
        return avg_loss

    def validation_step(self, graph_batch_dict, batch_idx, dataloader_idx=0):
        dataloader_tag = self.get_validation_dataloader_tag(dataloader_idx)
        with torch.no_grad():
            loss = self.training_step(graph_batch_dict)
            self.log(f"{dataloader_tag}/loss", loss)
            self._compute_and_update_accuracy_metrics(graph_batch_dict, dataloader_tag)
            return loss


class EmbeddingAndMaxMPNN(HamCycleFinderWithValueFunction):
    def _construct_embedding(self):
        embedding = ResidualMultilayerMPNN(self.hidden_dim, self.hidden_dim, 1, nr_layers=self.embedding_depth)
        embedding_out_projection = torch.nn.Linear(self.hidden_dim, self.hidden_dim - 3)
        return embedding, embedding_out_projection

    def _construct_processor(self):
        processor = ResidualMultilayerMPNN(self.hidden_dim, self.hidden_dim, 1, nr_layers=self.processor_depth)
        processor_out_projection = torch.nn.Linear(self.hidden_dim, self.out_dim)
        return processor, processor_out_projection

    def __init__(self, in_dim=3, out_dim=2, hidden_dim=32, embedding_depth=5, processor_depth=5,
                 value_function_weight=1, l2_regularization_weight=0.01, nr_simultaneous_simulations=8,
                 loss_type="mse", graph_updater_class=WalkUpdater, solution_scorer_class=scorers.SizeIndependentCombinatorialScorer, **kwargs):
        super().__init__(graph_updater_class, **kwargs)
        self.save_hyperparameters()

        self.scorer = solution_scorer_class()
        self.l2_regularization_weight = l2_regularization_weight
        self.value_function_weight = value_function_weight
        self.loss_type = loss_type
        self.nr_simultaneous_simulations=nr_simultaneous_simulations

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.embedding_depth = embedding_depth
        self.processor_depth = processor_depth
        assert hidden_dim > 3
        self.hidden_dim = hidden_dim
        self.softmax = torch.nn.Softmax(dim=0)
        self.initial_embedding = torch.nn.Parameter(torch.rand([hidden_dim], requires_grad=True))
        self.embedding, self.embedding_out_projection = self._construct_embedding()
        self.processor, self.processor_out_projection = self._construct_processor()

    def embed(self, d: torch_g.data.Data()):
        d.emb = self.embedding_out_projection(
            self.embedding.forward(torch.unsqueeze(self.initial_embedding, 0).expand(d.num_nodes, -1),
                                   d.edge_index, d.edge_attr))

    def forward(self, d: torch_g.data.Data) -> torch.Tensor:
        return self.processor_out_projection(
            torch.cat([d.emb, d.x], dim=1) + self.processor(torch.cat([d.emb, d.x], dim=1), d.edge_index, d.edge_attr))

    def raw_next_step_logits(self, d: torch_g.data.Data) -> torch.Tensor:
        return self.forward(d)[..., 0]

    def raw_next_step_logits_and_value_function(self, d: torch_g.data.Data) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.forward(d)
        logits = result[..., 0]
        value_estimate = torch.max(result[..., 1], -1)[0]
        if d.x[..., 0].any():
            return DataUtils._mask_neighbor_logits(logits, d), value_estimate
        else:
            return logits, value_estimate

    def init_graph(self, d) -> None:
        d.edge_attr = torch.ones([d.edge_index.shape[1], 1])
        self.embed(d)

    def training_step(self, simulations_dict):
        batched_simulation_states = BatchedSimulationStates.from_lightning_dict(simulations_dict)
        states = batched_simulation_states.batch_graph.to_data_list()
        actions = batched_simulation_states.actions
        rewards = batched_simulation_states.rewards
        simulation_depth = batched_simulation_states.simulation_depths

        batch_losses = []
        for (s, a, r) in zip(states, actions, rewards):
            self.init_graph(s) # TODO would be nice pass this through dataloaders and have it executed only on first step
            batch_losses.append(self._compute_loss(s, a , r))
        loss = torch.stack(batch_losses).mean()
        self.log_dict({
            "train/loss": loss,
            "train/avg_reward": torch.stack(rewards).mean(),
            "train/avg_simulation_depth": torch.stack(simulation_depth).type(torch.float32).mean()
            })
        return loss

    def on_validation_start(self) -> None:
        self._validation_nr_different_graphs_seen = 0
        return super().on_validation_start()

    def validation_step(self, simulations_dict, batch_ids, dataloader_idx):
        with torch.no_grad():
            loss = self.training_step(simulations_dict)
            self.log("validation/loss", loss)

            graphs = [graph for graph in BatchedSimulationStates.from_lightning_dict(simulations_dict).batch_graph.to_data_list() if len(DataUtils._starting_indices(graph)) == 0]
            if len(graphs) > 0:
                self._validation_nr_different_graphs_seen += len(graphs)
                batch_example = GraphBatchExample.from_graph_examples_list([GraphExample(graph, None) for graph in graphs])
                self._compute_and_update_accuracy_metrics(batch_example.to_lightning_dict(), self.validation_accuracy_metrics)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log_validation_accuracy_metrics()

    def create_simulation_batch(self, d):
        return torch_g.data.Batch.from_data_list([d.detach().clone() for _ in range(self.nr_simultaneous_simulations)])

    def _batch_simulate(self, graph : torch_g.data.Data, max_simulation_depth=-1):
        batch_graph = self.create_simulation_batch(graph)

        class SimulationRunInstructions(self._RunInstructions):
            def __init__(self, gnn_model: EmbeddingAndMaxMPNN, scorer) -> None:
                self.gnn_model = gnn_model
                self.scorer = scorer
                self.simulation_states = []

            def on_algorithm_start_fn(self, batch_graph):
                self.gnn_model.init_graph(batch_graph)
                self.gnn_model.prepare_for_first_step(batch_graph, [])
                self.states = []
                self.actions = []
                self.rewards = []

                start_step_number = 0
                return start_step_number

            def choose_next_step_fn(self, batch_graph, current_nodes, step_number, is_algorithm_stopped_mask):
                logits = self.gnn_model.next_step_logits_masked_over_neighbors(batch_graph).reshape([self.gnn_model.nr_simultaneous_simulations, -1])
                q = torch.distributions.Categorical(logits=logits[torch.logical_not(is_algorithm_stopped_mask)])
                choices = torch.zeros_like(logits[:,0], dtype=torch.long)
                choices[torch.logical_not(is_algorithm_stopped_mask)] = q.sample()
                choices += batch_graph.num_nodes // batch_graph.num_graphs * torch.arange(batch_graph.num_graphs)
                choices[is_algorithm_stopped_mask] = -1

                self.states += [batch_graph.clone()]
                self.actions += [torch.logical_not(is_algorithm_stopped_mask)* choices + (-1) * is_algorithm_stopped_mask.type(choices.dtype)]
                self.rewards += [self.scorer.batch_reward(batch_graph, choices, torch.logical_not(is_algorithm_stopped_mask))]
                return choices

        simulation_run_instructions = SimulationRunInstructions(self, self.scorer)
        with torch.no_grad():
            self._run_on_graph_batch(batch_graph, simulation_run_instructions)

        state_batches, actions_batches, reward_batches = simulation_run_instructions.states, simulation_run_instructions.actions, simulation_run_instructions.rewards
        for i in range(len(reward_batches) - 2, -1, -1):
            reward_batches[i] += reward_batches[i + 1]
        return state_batches, actions_batches, reward_batches

    def _compute_loss(self, state, action, reward):
        logits, value_estimate = self.raw_next_step_logits_and_value_function(state)
        q = torch.distributions.Categorical(logits=logits)
        l2_params = torch.sum(torch.stack([torch.sum(torch.square(p)) for p in self.parameters()]))
        value_estimate_loss = torch.square(reward - value_estimate)
        REINFORCE_loss = -(reward - value_estimate).detach() * q.log_prob(action)
        total_loss = REINFORCE_loss + self.l2_regularization_weight * l2_params + self.value_function_weight * value_estimate_loss
        return total_loss

    def _run_episode(self, original_graph: torch_g.data.Data):
        d = original_graph.clone()
        state_batches, action_batches, reward_batches = \
            self._batch_simulate(d)

        states = []
        for batch in state_batches:
            for state_index, state in enumerate(batch.to_data_list()):
                state.x = batch.x[original_graph.num_nodes * state_index: original_graph.num_nodes * (state_index + 1)]
                states.append(state)

        actions = [action if action > -1 else None for batch in action_batches for action in
                   (batch - d.num_nodes * torch.arange(batch.shape[0]))] # TODO quicfix compile code
        rewards = [reward for batch in reward_batches for reward in batch]

        actions_tensor = torch.stack(action_batches)
        step_taken_tensor = (actions_tensor > -1).type(torch.int64)
        simulation_tensor = step_taken_tensor.flip(0).cumsum(0).flip(0)
        simulation_depths = [x for x in simulation_tensor.flatten()]

        simulation_data = [
            SimulationState(state, action, reward, depth)
            for state, action, reward, depth in zip(states, actions, rewards, simulation_depths)
            if action is not None
        ]
        return simulation_data

    def description(self):
        return f"embedding (hidden dim={self.hidden_dim}): {torchinfo.summary(self.embedding, verbose=0, depth=5)}\n" \
               f"processor (hidden dim={self.hidden_dim}): {torchinfo.summary(self.processor, verbose=0, depth=5)}"


class GatedGCNEmbedAndProcess(EmbeddingAndMaxMPNN):
    def _construct_embedding(self):
        embedding = MultilayerGatedGCN(self.hidden_dim, self.embedding_depth)
        embedding_out_projection = torch.nn.Linear(self.hidden_dim, self.hidden_dim - 3)
        return embedding, embedding_out_projection

    def _construct_processor(self):
        processor = MultilayerGatedGCN(self.hidden_dim, self.processor_depth)
        processor_out_projection = torch.nn.Linear(self.hidden_dim, self.out_dim)
        return processor, processor_out_projection

    def embed(self, d: torch_g.data.Data()):
        emb, d.edge_attr = self.embedding.forward(torch.squeeze(self.initial_embedding, 0).expand(d.num_nodes, -1),
                                                  d.edge_index, d.edge_attr.squeeze(-1))
        d.emb = self.embedding_out_projection(emb)

    def forward(self, d: torch_g.data.Data) -> torch.Tensor:
        x, e_hat = self.processor(torch.cat([d.emb, d.x], dim=1), d.edge_index, d.edge_attr)
        return self.processor_out_projection(x)

    def init_graph(self, d) -> None:
        d.edge_attr = torch.ones([d.edge_index.shape[1], self.hidden_dim]) # TODO this is duplicated from superclass (also, would be nice to only do it once and pass embeding through dataloaders)
        self.embed(d)
