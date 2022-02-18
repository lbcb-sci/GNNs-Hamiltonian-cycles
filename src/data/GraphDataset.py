from collections import deque
import torch
import torch.nn.functional as F
import torch_geometric as torch_geometric
from typing import List
import random


def get_shifts_for_graphs_in_batch(batch_graph: torch_geometric.data.Batch):
    batch_graph_one_hot = F.one_hot(batch_graph.batch, batch_graph.num_graphs)
    graph_sizes = torch.sum(batch_graph_one_hot, dim=0)
    graph_shifts = graph_sizes.cumsum(0).roll(1)
    graph_shifts[0] = 0
    return graph_shifts


class GraphExample:
    def __init__(self, graph:torch_geometric.data.Data, teacher_path: torch.Tensor, teacher_distribution: torch.Tensor=None) -> None:
        self.graph = graph
        self.teacher_path = teacher_path
        self.teacher_distribution = teacher_distribution

class GraphBatchExample:
    def __init__(self, graph_batch: torch_geometric.data.Batch, teacher_paths: List[torch.Tensor], teacher_distributions: torch.Tensor=None) -> None:
        self.graph_batch = graph_batch
        self.teacher_paths = teacher_paths
        self.teacher_distributions = teacher_distributions

    @staticmethod
    def from_graph_examples_list(graph_examples):
        graphs = [example.graph for example in graph_examples]
        batch_graph = torch_geometric.data.Batch.from_data_list(graphs)
        graph_shifts = get_shifts_for_graphs_in_batch(batch_graph)
        return GraphBatchExample(
            batch_graph,
            [example.teacher_path + shift for example, shift in zip(graph_examples, graph_shifts)],
            [example.teacher_distribution for example in graph_examples])

    def to_lightning_dict(self):
        # TODO see if pytorch_lighting can be adjusted to move custom classes from and to devices.
        # It handles dictionaries of tensors without problems
        graphs = self.graph_batch.to_data_list()
        return {
            "list_of_edge_indexes": [graph.edge_index for graph in graphs],
            "list_of_graph_num_nodes": [graph.num_nodes for graph in graphs],
            "teacher_paths": self.teacher_paths,
        }

class SimulationState:
    def __init__(self, graph_state: torch_geometric.data.Data, action: int, reward: int, simulation_depth: int) -> None:
        self.graph_state = graph_state
        self.action = action
        self.reward = reward
        self.simulation_depth = simulation_depth

class BatchedSimulationStates:
    def __init__(self, batch_graph, actions, rewards, simulation_depths) -> None:
        self.batch_graph = batch_graph
        self.actions = actions
        self.rewards = rewards
        self.simulation_depths = simulation_depths

    @staticmethod
    def from_simulation_states_list(graph_simulation_examples: list[SimulationState]):
        graphs = [example.graph_state for example in graph_simulation_examples]
        return BatchedSimulationStates(
            batch_graph = torch_geometric.data.Batch.from_data_list(graphs),
            actions = [example.action for example in graph_simulation_examples],
            rewards = [example.reward for example in graph_simulation_examples],
            simulation_depths = [example.simulation_depth for example in graph_simulation_examples]
        )

    def to_lightning_dict(self):
        graphs = self.batch_graph.to_data_list()
        return {
            "list_of_edge_indices": [graph.edge_index for graph in graphs],
            "list_of_graph_num_nodes": [graph.num_nodes for graph in graphs],
            "list_of_per_node_markers": [graph.x for graph in graphs],
            "actions": self.actions,
            "rewards": self.rewards,
            "simulation_depths": self.simulation_depths
        }

    @staticmethod
    def from_lightning_dict(lightning_dict):
        list_of_edge_indices = lightning_dict["list_of_edge_indices"]
        list_of_graph_num_nodes = lightning_dict["list_of_graph_num_nodes"]
        list_of_graph_per_node_x = lightning_dict["list_of_per_node_markers"]
        graph_list  = []
        for edge_index, num_nodes, x in zip(list_of_edge_indices, list_of_graph_num_nodes, list_of_graph_per_node_x):
            graph = torch_geometric.data.Data(edge_index=edge_index, num_nodes=num_nodes)
            graph.x = x
            graph_list.append(graph)
        batch_graph = torch_geometric.data.Batch.from_data_list(graph_list)
        return BatchedSimulationStates(batch_graph, lightning_dict["actions"], lightning_dict["rewards"], lightning_dict["simulation_depths"])


class GraphGeneratingDataset(torch.utils.data.Dataset):
    def __init__(self, graph_generator, virtual_epoch_size=1000, ) -> None:
        super().__init__()
        self.graph_generator = iter(graph_generator)
        self.virtual_epoch_size = virtual_epoch_size

    def __len__(self):
        return self.virtual_epoch_size

    def __getitem__(self, idx):
        generated_graph_example = next(self.graph_generator)
        return generated_graph_example


class GraphDataLoader(torch.utils.data.DataLoader):
    @staticmethod
    def graph_collate_fn(graph_examples: List[GraphExample]) -> dict:
        graph_batch_example = GraphBatchExample.from_graph_examples_list(graph_examples)
        return graph_batch_example.to_lightning_dict()

    def __init__(self, dataset: GraphGeneratingDataset, *args, **kwargs) -> None:
        super().__init__(dataset, collate_fn=self.graph_collate_fn, *args, **kwargs)


#TODO typehint EmbeddProcessDatamodule, circular references are the problem
class SimulationsDataset(torch.utils.data.DataLoader):
    def __init__(self, graph_generator, simulation_module_reference, virtual_epoch_size=1000, update_simulation_module_after_b_steps=1, nr_simultaneous_simulations=1) -> None:
        self.graph_generator = graph_generator
        self.original_module = simulation_module_reference
        self.virtual_epoch_size = virtual_epoch_size
        self.update_simulation_module_after_n_steps = update_simulation_module_after_b_steps
        self.nr_simulatneous_simulations = nr_simultaneous_simulations
        self.simulations_count = 0
        self.simulations_storage = deque()
        self._create_module_copy_for_simulation()
        self._update_weights_of_simulation_model()

    def _update_weights_of_simulation_model(self):
        for original_param, for_simulation_param in zip(self.original_module.parameters(), self._module_for_simulations.parameters()):
            for_simulation_param.data.copy_(original_param).detach_()

    def _create_module_copy_for_simulation(self):
        self._module_for_simulations = self.original_module.__class__(**self.original_module.hparams)

    def __len__(self):
        return self.virtual_epoch_size

    def __getitem__(self, idx):
        if self.simulations_storage:
            return self.simulations_storage.pop()

        if self.simulations_count >= self.update_simulation_module_after_n_steps:
            self.simulations_count = 0
            self._update_weights_of_simulation_model()

        simulation_data = self._module_for_simulations._run_episode(next(iter(self.graph_generator)))
        random.shuffle(simulation_data)
        self.simulations_storage.extend(simulation_data)
        self.simulations_count += 1
        return self.simulations_storage.pop()

class SimulationStatesDataLoader(torch.utils.data.DataLoader):
    @staticmethod
    def graph_collate_fn(graph_state_examples: List[SimulationState]) -> dict:
        batched_simulation_states = BatchedSimulationStates.from_simulation_states_list(graph_state_examples)
        return batched_simulation_states.to_lightning_dict()

    def __init__(self, dataset: GraphGeneratingDataset, *args, **kwargs) -> None:
        super().__init__(dataset, collate_fn=self.graph_collate_fn, *args, **kwargs)