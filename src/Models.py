import itertools
import os.path
from abc import ABC, abstractmethod
from typing import List
import numpy
import copy

import torch
import torch.nn.functional as F
import torch_scatter
import torch_geometric as torch_g
import torchinfo
import pytorch_lightning as torch_lightning
import torchmetrics

from src.HamiltonSolver import HamiltonSolver
from src.NN_modules import ResidualMultilayerMPNN, MultilayerGatedGCN
from src.data.GraphDataset import BatchedSimulationStates, SimulationState
from src.constants import MODEL_WEIGHTS_FOLDER
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
    def update_state(d: torch_g.data.Data, current_nodes):
        previous = torch.squeeze(torch.nonzero(torch.eq(d.x[..., 1], torch.ones_like(d.x[..., 1]))), -1)

        if torch.any(current_nodes < 0).item() or torch.any(current_nodes > d.num_nodes).item():
            raise Exception("Illegal choice of next step node when updating graph mask!")

        d.x[previous, 1] = 0
        d.x[current_nodes, 1] = 1
        d.x[current_nodes, 2] = 1


class HamFinderGNN(HamiltonSolver, torch_lightning.LightningModule):
    def __init__(self, graph_updater: WalkUpdater):
        super(HamFinderGNN, self).__init__()
        self.graph_updater = graph_updater
        self.BATCH_SIZE_DURING_INFERENCE = 8

        self.validation_accuracy_metrics, self.test_accuracy_metrics = \
            [torch.nn.ModuleDict({
                name: torchmetrics.Accuracy(compute_on_step=False)
                for name in ("hamiltonian_cycle", "hamiltonian_path", "90_perc_cycle", "90_perc_path")}
                                 ) for _ in range(2)]

    @abstractmethod
    def next_step_logits(self, d: torch_g.data.Batch) -> torch.Tensor:
        pass

    def next_step_prob(self, d: torch_g.data.Batch) -> torch.Tensor:
        logits = self.next_step_logits(d)
        p = torch_scatter.scatter_softmax(logits, d.batch)
        return p

    @staticmethod
    def _neighbor_indices(d: torch_g.data.Data):
        current = torch.nonzero(torch.isclose(d.x[..., 1], torch.ones_like(d.x[..., 1]))).squeeze(-1)
        if current.numel == 0:
            return []
        neighbor_index = d.edge_index[1, torch.any(d.edge_index[None, 0, :] == current[:, None], dim=0)]
        return neighbor_index.unique()

    @staticmethod
    def _mask_neighbor_logits(logits, d: torch_g.data.Data):
        valid_next_step_indices = HamFinderGNN._neighbor_indices(d)
        neighbor_logits = torch.zeros_like(logits).log()
        neighbor_logits[valid_next_step_indices] = logits.index_select(0, valid_next_step_indices)
        return neighbor_logits

    def next_step_logits_masked_over_neighbors(self, d: torch_g.data.Batch) -> torch.Tensor:
        if torch.all(torch.lt(d.x[..., 1], 0.5)):
            return self.next_step_logits(d)
        logits = self.next_step_logits(d)
        return self._mask_neighbor_logits(logits, d)

    def next_step_prob_masked_over_neighbors(self, d: torch_g.data.Batch) -> torch.Tensor:
        logits = self.next_step_logits_masked_over_neighbors(d)
        neighbor_prob = torch_scatter.scatter_softmax(logits, d.batch)
        return neighbor_prob

    def prepare_for_first_step(self, d: torch_g.data.Batch, start_batch):
        return self.graph_updater.batch_prepare_for_first_step(d, start_batch)

    def update_state(self, d: torch_g.data.Data, current_batch):
        return self.graph_updater.update_state(d, current_batch)

    @abstractmethod
    def init_graph(self, d) -> None:
        pass

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def load_weights(self, out_folder):
        pass

    @abstractmethod
    def save_weights(self, out_folder):
        pass

    def _neighbor_prob_and_greedy_choice_for_batch(self, d: torch_g.data.Batch):
        p = self.next_step_prob_masked_over_neighbors(d)
        graph_sizes = [g.num_nodes for g in d.to_data_list()]
        max_size = max(graph_sizes)
        p = torch.stack([F.pad(x, (0, max_size - x.shape[0]), "constant", -1) for x in torch.split(p, graph_sizes)])

        choice = torch.argmax(
            torch.isclose(p, torch.max(p, dim=-1)[0][..., None])
            * (p + torch.randperm(p.shape[-1], device=p.device)[None, ...]), dim=-1)
        choice += torch.tensor(list(itertools.accumulate(graph_sizes[:-1], initial=0)))

        return p, choice

    def batch_run_greedy_neighbor(self, batch_data: torch_g.data.Batch):
        with torch.no_grad():
            nodes_per_graph = [d.num_nodes for d in batch_data.to_data_list()]
            max_steps_in_a_cycle = max(nodes_per_graph) + 1

            self.init_graph(batch_data)
            all_choices, all_scores = [], []
            choice = None
            stop_algorithm_mask = torch.zeros(
                [batch_data.num_graphs], device=batch_data.edge_index.device, dtype=torch.uint8)

            for step in range(0, max_steps_in_a_cycle):
                if step < 2:
                    self.prepare_for_first_step(batch_data, choice)
                else:
                    current_nodes = choice
                    self.update_state(batch_data, current_nodes[current_nodes != -1])
                p, choice = self._neighbor_prob_and_greedy_choice_for_batch(batch_data)
                choice = torch.logical_not(stop_algorithm_mask) * choice - stop_algorithm_mask
                p = p * torch.logical_not(stop_algorithm_mask)[..., None] - stop_algorithm_mask[..., None]
                if len(all_choices) > 0:
                    already_visited_mask = torch.any(
                        torch.stack([torch.eq(x, choice) for x in all_choices], -1), -1)
                    stop_algorithm_mask = torch.maximum(stop_algorithm_mask, already_visited_mask)
                all_choices.append(choice)
                all_scores.append(p)

                if torch.all(stop_algorithm_mask).item():
                    break
            walks = torch.stack(all_choices, -1)
            selections = torch.stack(all_scores, -2)
            if walks.shape[-1] != max_steps_in_a_cycle:
                walks = F.pad(walks, (0, max_steps_in_a_cycle - walks.shape[-1]), value=-1)
            if selections.shape[-2] != max_steps_in_a_cycle:
                selections = F.pad(selections, (0, 0, 0, max_steps_in_a_cycle - selections.shape[-2]), value=-1)
            return walks, selections

    def get_batch_size_for_multi_solving(self):
        return self.BATCH_SIZE_DURING_INFERENCE

    @staticmethod
    def _conver_batch_walk_tensor_into_solution_list(batch_walks_tensor, graphs_shift_inside_batch):
        batch_walks_tensor -= graphs_shift_inside_batch[:, None]
        batch_walks_tensor[batch_walks_tensor < 0] = -1
        raw_walks = [[int(node.item()) for node in walk] for walk in batch_walks_tensor]
        walks = []
        for walk in raw_walks:
            walk_end_index = walk.index(-1) if -1 in walk else len(walk)
            walks.append(walk[:walk_end_index])
        return walks

    @staticmethod
    def _unpack_graph_batch_dict(graph_batch_dict) -> tuple[torch_g.data.Batch, List[List[int]]]:
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
        walks_tensor, _ = self.batch_run_greedy_neighbor(batch_graph)
        return self._conver_batch_walk_tensor_into_solution_list(walks_tensor, batch_shift)

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

    def update_accuracy_metrics(self, accuracy_metrics_dict: dict[str, torchmetrics.Accuracy], graphs: list[torch_g.data.Data], solutions: list[list[int]]):
        evals = Evaluation.EvaluationScores.evaluate(graphs, solutions)
        df_scores = Evaluation.EvaluationScores.compute_scores(evals)
        for metric_name, column_name in [("hamiltonian_cycle", "is_ham_cycle"), ("hamiltonian_path", "is_ham_path"),
                                         ("90_perc_cycle", "is_approx_ham_cycle"), ("90_perc_path", "is_approx_ham_path")]:
            if metric_name in accuracy_metrics_dict:
                metric = accuracy_metrics_dict[metric_name]
                prediction = torch.tensor(df_scores[column_name], dtype=torch.float)
                metric.update(prediction, torch.ones_like(prediction, dtype=torch.int))


class HamCycleFinderWithValueFunction(HamFinderGNN):
    @abstractmethod
    def next_step_logits_and_value_function(self, d:torch_g.data.Data) -> [torch.Tensor, torch.Tensor]:
        pass


class EncodeProcessDecodeAlgorithm(HamFinderGNN):
    def _construct_processor(self):
        return ResidualMultilayerMPNN(dim=self.hidden_dim, message_dim=self.hidden_dim, edges_dim=1,
                                      nr_layers=self.processor_depth)

    def _construct_encoder_and_decoder(self):
        encoder_nn = torch.nn.Sequential(torch.nn.Linear(3 + self.hidden_dim, self.hidden_dim))
        decoder_nn = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim + self.hidden_dim, self.out_dim))
        return encoder_nn, decoder_nn

    def __init__(self, is_load_weights=False, processor_depth=3, in_dim=1, out_dim=1, hidden_dim=32, graph_updater=WalkUpdater(), loss_type="mse"):
        super().__init__(graph_updater)

        self.save_hyperparameters()

        self.PROCESSOR_NAME = f"{self.__class__.__name__}_Processor.tar"
        self.ENCODER_NAME = f"{self.__class__.__name__}_Encoder.tar"
        self.DECODER_NAME = f"{self.__class__.__name__}_Decoder.tar"
        self.INITIAL_HIDDEN_TENSOR_NAME = f"{self.__class__.__name__}_InitialHiddenTensor.pt"

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.processor_depth = processor_depth
        assert loss_type in ["mse", "entropy"]
        self.loss_type = loss_type

        self.encoder_nn, self.decoder_nn = self._construct_encoder_and_decoder()
        self.processor_nn = self._construct_processor()
        self.initial_h = torch.nn.Parameter(torch.rand(self.hidden_dim))

        if is_load_weights:
            self.load_weights()


    # def on_load_checkpoint(self, checkpoint) -> None:
    #     _processor_depth_varname = "processor_depth"
    #     if _processor_depth_varname in checkpoint:
    #         self.processor_depth = checkpoint[_processor_depth_varname]
    #         self._construct_processor()


    def description(self):
        return f"encoder: {torchinfo.summary(self.encoder_nn, verbose=0, depth=5)}\n" \
               f"processor (hidden dim={self.hidden_dim}):" \
               f" {torchinfo.summary(self.processor_nn, verbose=0, depth=5)}\n" \
               f"decoder: {torchinfo.summary(self.decoder_nn, verbose=0, depth=5)}"

    def get_weights_paths(self, directory):
        return [os.path.join(directory, name) for name in
                [self.ENCODER_NAME, self.DECODER_NAME, self.PROCESSOR_NAME, self.INITIAL_HIDDEN_TENSOR_NAME]]

    def load_weights(self, directory=MODEL_WEIGHTS_FOLDER):
        encoder_path, decoder_path, processor_path, initial_hidden_path = self.get_weights_paths(directory)
        for module, path in zip(
                [self.encoder_nn, self.decoder_nn, self.processor_nn],
                [encoder_path, decoder_path, processor_path]):
            if os.path.isfile(path):
                module.load_state_dict(torch.load(path))
                print("Loaded {} from {}".format(module.__class__.__name__, path))
        if os.path.isfile(initial_hidden_path):
            self.initial_h = torch.nn.Parameter(torch.load(initial_hidden_path))
            # self.initial_h = self.initial_h.to(self.device)
        else:
            self.initial_h = torch.nn.Parameter(torch.rand(self.hidden_dim, device=self.device))

    def parameters(self):
        return itertools.chain(self.encoder_nn.parameters(), self.decoder_nn.parameters(),
                               self.processor_nn.parameters(), [self.initial_h])

    def next_step_logits(self, d: torch_g.data.Data):
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

        self.init_graph(batch_graph)
        self.prepare_for_first_step(batch_graph, None)
        self.next_step_logits(batch_graph)
        self.prepare_for_first_step(batch_graph, [teacher_path[0] for teacher_path in teacher_paths])

        loss = torch.zeros(1, device=batch_graph.edge_index.device)
        graph_sizes = torch.sum(F.one_hot(batch_graph.batch, batch_graph.num_graphs), dim=0)
        graph_shifts = graph_sizes.cumsum(0).roll(1)
        graph_shifts[0] = 0
        max_steps = torch.max(graph_sizes, dim=-1).values + 1
        for step in range(1, max_steps):
            active_graphs_index = [index for index in range(len(teacher_paths)) if len(teacher_paths[index]) > step]
            active_teacher_paths = [teacher_paths[index] for index in active_graphs_index]
            current_nodes, next_step_nodes = [torch.stack([teacher_path[node_index] for teacher_path in active_teacher_paths])
                                              for node_index in (step-1, step)]
            if step > 1:
                self.update_state(batch_graph, current_nodes)

            if self.loss_type == "mse":
                p = self.next_step_prob(batch_graph)
            elif self.loss_type == "entropy":
                logits = self.next_step_logits(batch_graph)

            subgraph_losses = []
            for graph_index in range(len(graph_shifts)):
                graph_start_index = graph_shifts[graph_index]
                graph_end_index = graph_shifts[graph_index+1] if graph_index + 1 < len(graph_shifts) else batch_graph.num_nodes
                if self.loss_type == "mse":
                    mse_loss = torch.nn.MSELoss()
                    _graph_p = p[graph_start_index: graph_end_index]
                    _teacher_p = torch.zeros_like(_graph_p)
                    _teacher_p[next_step_nodes[graph_index] - graph_start_index] = 1.
                    loss += mse_loss(_graph_p, _teacher_p)
                elif self.loss_type == "entropy":
                    entropy_loss = torch.nn.CrossEntropyLoss()
                    _graph_logits = logits[graph_start_index: graph_end_index].unsqueeze(0)
                    subgraph_losses.append(entropy_loss(_graph_logits, next_step_nodes[graph_index:graph_index + 1] - graph_start_index))
            loss += torch.stack(subgraph_losses).sum()
        loss = loss / (max_steps * len(graph_sizes))
        self.log("train/loss", loss)
        return loss

    def validation_step(self, graph_batch_dict, dataloader_idx):
        with torch.no_grad():
            batch_graph, _ = self._unpack_graph_batch_dict(graph_batch_dict)
            loss = self.training_step(graph_batch_dict)
            self.log("validation/loss", loss)
            solutions = self.solve_batch_graph(batch_graph)
            self.update_accuracy_metrics(self.validation_accuracy_metrics, batch_graph.to_data_list(), solutions)
            return loss

    def on_validation_epoch_end(self) -> None:
        for metric_name, metric in self.validation_accuracy_metrics.items():
            self.log(f"validation/{metric_name}", metric.compute())
            metric.reset()
        return super().on_validation_epoch_end()

    def predict_step(self, graph_batch_dict, batch_idx):
        batch_graph, _ = self._unpack_graph_batch_dict(graph_batch_dict)
        walk_tensors = self.solve_batch_graph(batch_graph)
        return walk_tensors

    def test_step(self, graph_batch_dict, batch_idx, dataloader_idx=None):
        batch_graph, _ = self._unpack_graph_batch_dict(graph_batch_dict)
        walks = self.predict_step(graph_batch_dict, batch_idx)
        graph_list = batch_graph.to_data_list()
        self.update_accuracy_metrics(self.test_accuracy_metrics, graph_list, walks)

    def on_test_epoch_end(self) -> None:
        for metric_name, metric in self.test_accuracy_metrics.items():
            self.log(f"test/{metric_name}", metric.compute())
            metric.reset()
        return super().on_test_epoch_end()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-4)

    def save_weights(self, directory=MODEL_WEIGHTS_FOLDER):
        encoder_path, decoder_path, processor_path, initial_hidden_path = self.get_weights_paths(directory)
        for module, path in zip(
                [self.processor_nn, self.encoder_nn, self.decoder_nn],
                [processor_path, encoder_path, decoder_path]):
            torch.save(module.state_dict(), path)
        torch.save(self.initial_h, initial_hidden_path)


class EmbeddingAndMaxMPNN(HamCycleFinderWithValueFunction):
    def _construct_embedding(self):
        embedding = ResidualMultilayerMPNN(self.hidden_dim, self.hidden_dim, 1, nr_layers=self.embedding_depth)
        embedding_out_projection = torch.nn.Linear(self.hidden_dim, self.hidden_dim - 3)
        return embedding, embedding_out_projection

    def _construct_processor(self):
        processor = ResidualMultilayerMPNN(self.hidden_dim, self.hidden_dim, 1, nr_layers=self.processor_depth)
        processor_out_projection = torch.nn.Linear(self.hidden_dim, self.out_dim)
        return processor, processor_out_projection

    def __init__(self, is_load_weights=True, in_dim=3, out_dim=2, hidden_dim=32, embedding_depth=5, processor_depth=5,
                 value_function_weight=1, l2_regularization_weight=0.01,
                 loss_type="mse", graph_updater=WalkUpdater(), solution_scorer=scorers.CombinatorialScorer()):
        super().__init__(graph_updater)
        self.save_hyperparameters()

        self.scorer = solution_scorer
        self.l2_regularization_weight = l2_regularization_weight
        self.value_function_weight = value_function_weight

        self.EMBEDDING_NAME = "{}-Embedding.tar".format(self.__class__.__name__)
        self.PROCESSOR_NAME = "{}-Processor.tar".format(self.__class__.__name__)

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

        if is_load_weights:
            self.load_weights()

    # def to(self, device):
    #     for module in [self.embedding, self.embedding_out_projection, self.processor, self.processor_out_projection]:
    #         module.to(device)
    #     self.initial_embedding = self.initial_embedding.to(device)
    #     self.device = device

    # def to_cuda(self):
    #     self.initial_embedding.to("cuda")
    #     for module in [self.embedding, self.embedding_out_projection, self.processor, self.processor_out_projection]:
    #         module.cuda()
    #     self.device = "cuda"

    def embed(self, d: torch_g.data.Data()):
        d.emb = self.embedding_out_projection(
            self.embedding.forward(torch.unsqueeze(self.initial_embedding, 0).expand(d.num_nodes, -1),
                                   d.edge_index, d.edge_attr))

    def forward(self, d: torch_g.data.Data) -> torch.Tensor:
        return self.processor_out_projection(
            torch.cat([d.emb, d.x], dim=1) + self.processor(torch.cat([d.emb, d.x], dim=1), d.edge_index, d.edge_attr))

    def next_step_logits(self, d: torch_g.data.Data) -> torch.Tensor:
        return self.forward(d)[..., 0]

    def next_step_logits_and_value_function(self, d: torch_g.data.Data) -> [torch.Tensor, torch.Tensor]:
        result = self.forward(d)
        logits = result[..., 0]
        value_estimate = torch.max(result[..., 1], -1)[0]
        if d.x[..., 0].any():
            return HamFinderGNN._mask_neighbor_logits(logits, d), value_estimate
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

        loss = torch.zeros([1], device=states[0].edge_index.device)
        for (s, a, r) in zip(states, actions, rewards):
            self.init_graph(s) # TODO would be nice pass this through dataloaders and have it executed only on first step
            loss += self._compute_loss(s, a , r)
        self.log("train/avg_reward", torch.stack(rewards).mean())
        return loss


    def _batch_simulate(self, batched_graph: torch_g.data.Data, max_simulation_depth=-1):
        # TODO tmp compilation code
        d = batched_graph.to_data_list()[0]
        simulation_batch_size = 1

        max_simulation_steps = d.num_nodes + 1
        if max_simulation_depth >= 1:
            max_simulation_steps = min(max_simulation_steps, max_simulation_depth)
        with torch.no_grad():
            d = torch_g.data.Batch.from_data_list([d]) # TODO tmp compilation code
            simulation_running_flag = torch.ones([d.num_graphs], dtype=torch.bool, device=d.x.device)
            states, actions, rewards = [], [], []
            step_no = 0
            while simulation_running_flag.any() and step_no < max_simulation_steps + 1:
                step_no += 1
                logits = self.next_step_logits_masked_over_neighbors(d).reshape([simulation_batch_size, -1])
                q = torch.distributions.Categorical(logits=logits)
                choices = q.sample() + d.num_nodes // d.num_graphs * torch.arange(d.num_graphs)
                states += [d.clone()]
                actions += [simulation_running_flag * choices + (-1) * torch.logical_not(simulation_running_flag)]
                rewards += [self.scorer.batch_reward(d, choices, simulation_running_flag)]
                choices_mask = torch.zeros_like(d.x[..., 2]).scatter_(0, choices, 1).reshape([d.num_graphs, -1])
                simulation_running_flag = torch.minimum(
                    simulation_running_flag, torch.logical_not(torch.minimum(
                        choices_mask, d.x[..., 2].reshape([d.num_graphs, -1])).any(-1)))
                if not d.x[..., 0].any():
                    self.prepare_for_first_step(d, choices)
                else:
                    self.update_state(d, choices)
            for i in range(len(rewards) - 2, -1, -1):
                rewards[i] += rewards[i + 1]
            return states, actions, rewards

    def _compute_loss(self, state, action, reward):
        logits, value_estimate = self.next_step_logits_and_value_function(state)
        q = torch.distributions.Categorical(logits=logits)
        l2_params = torch.sum(torch.stack([torch.sum(torch.square(p)) for p in self.parameters()]))
        value_estimate_loss = torch.square(reward - value_estimate)
        REINFORCE_loss = -(reward - value_estimate).detach() * q.log_prob(action)
        total_loss = REINFORCE_loss + self.l2_regularization_weight * l2_params + self.value_function_weight * value_estimate_loss
        return total_loss

    def _run_episode(self, original_graph: torch_g.data.Data):
        d = original_graph.clone()
        self.init_graph(d)
        self.prepare_for_first_step(d, None)
        state_batches, actions_batches, rewards_batches = \
            self._batch_simulate(torch_g.data.Batch.from_data_list([d]))

        states = [s for b in state_batches for s in b.to_data_list()]
        actions = [a if a > -1 else -1 for b in actions_batches for a in
                   (b - d.num_nodes * torch.arange(b.shape[0])).split(1)] # TODO quicfix compile code
        rewards = [r for b in rewards_batches for r in b.split(1)]

        simulation_depths = [step for step, batch in enumerate(state_batches) for _ in batch.to_data_list()]

        simulation_data = [
            SimulationState(state, action, reward, depth)
            for state, action, reward, depth in zip(states, actions, rewards, simulation_depths)
            if action.item() != -1
        ]
        return simulation_data

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-5)


    # def get_device(self) -> str:
    #     return self.device

    def parameters(self):
        return itertools.chain([self.initial_embedding], self.embedding.parameters(),
                               self.embedding_out_projection.parameters(), self.processor.parameters(),
                               self.processor_out_projection.parameters())

    def description(self):
        return f"embedding (hidden dim={self.hidden_dim}): {torchinfo.summary(self.embedding, verbose=0, depth=5)}\n" \
               f"processor (hidden dim={self.hidden_dim}): {torchinfo.summary(self.processor, verbose=0, depth=5)}"

    def get_weights_paths(self, directory):
        return [os.path.join(directory, name) for name in [self.EMBEDDING_NAME, self.PROCESSOR_NAME]]

    def load_weights(self, directory=MODEL_WEIGHTS_FOLDER):
        embedding_path, processor_path = self.get_weights_paths(directory)
        if os.path.isfile(embedding_path):
            embedding_save_data = torch.load(embedding_path)
            self.initial_embedding = torch.nn.Parameter(embedding_save_data["initial"]["initial"])
            for module_key, module in zip(["MPNN", "out_projection"], [self.embedding, self.embedding_out_projection]):
                module.load_state_dict(embedding_save_data[module_key])
            print("Loaded embedding module from {}".format(embedding_path))
        if os.path.isfile(processor_path):
            processor_save_data = torch.load(processor_path)
            for modul_key, module in zip(["MPNN", "out_projection"], [self.processor, self.processor_out_projection]):
                module.load_state_dict(processor_save_data[modul_key])
            print("Loaded processor module from {}".format(processor_path))

    def save_weights(self, directory=MODEL_WEIGHTS_FOLDER):
        embedding_path, processor_path = self.get_weights_paths(directory)
        embedding_save_data = {
            'initial': {"initial": self.initial_embedding},
            'MPNN': self.embedding.state_dict(),
            'out_projection': self.embedding_out_projection.state_dict()
        }
        processor_save_data = {
            'MPNN': self.processor.state_dict(),
            'out_projection': self.processor_out_projection.state_dict()
        }
        for save_data, path in zip([embedding_save_data, processor_save_data],
                                   [embedding_path, processor_path]):
            torch.save(save_data, path)


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



    # def __init__(self, nr_epochs, iterations_in_epoch, episodes_per_example=1, scorer=CombinatorialScorer(1, -1, 2, 1),
    #              l2_regularization_weight=0.01, value_function_weight=1,
    #              simulation_batch_size=8, verbose=1, max_simulation_steps=None):
    # HamR_trainer = REINFORCE_WithLearnableBaseline(nr_epochs=train_epochs, iterations_in_epoch=100,
    #                                                episodes_per_example=1, simulation_batch_size=1)
    #     return super().configure_optimizers()
