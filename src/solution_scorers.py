import torch
import torch.nn.functional as F
import torch_geometric as torch_g
from abc import ABC

from src.HamiltonSolver import DataUtils

class ReinforcementScorer(ABC):
    def batch_reward(self, d: torch_g.data.Batch, choices, simulation_running_flag=None) -> torch.Tensor:
        pass

class CombinatorialScorer(ReinforcementScorer):
    def __init__(self, valid_next_node_reward=1., illegal_next_node_multiplier=-1., complete_cycle_multiplier=2.,
                 complete_tour_multiplier=1.):
        self.valid_next_node_reward = valid_next_node_reward
        self.illegal_next_node_multiplier = illegal_next_node_multiplier
        self.complete_cycle_multiplier = complete_cycle_multiplier
        self.complete_tour_multiplier = complete_tour_multiplier

    def batch_reward(self, d: torch_g.data.Batch, choices, simulation_running_flag): # TODO zero step reward seems to be off
        num_nodes_per_graph = d.num_nodes // d.num_graphs
        valid_next_node_reward = self.valid_next_node_reward
        illegal_next_node_reward = self.illegal_next_node_multiplier * num_nodes_per_graph
        complete_cycle_reward = self.complete_cycle_multiplier * num_nodes_per_graph
        complete_tour_reward = self.complete_tour_multiplier * num_nodes_per_graph

        start_indices = DataUtils._starting_indices(d)
        neighbor_indices = DataUtils._neighbor_indices(d)
        if  len(start_indices) == 0:
            valid_next_step_mask = torch.ones_like(choices, dtype=torch.bool)
        elif len(start_indices) == d.num_graphs:
            valid_next_step_mask = torch.isin(choices, neighbor_indices)
        else:
            raise RuntimeError("Not implemented: Don't know how to reward batch where some the initial step needs to be selected on some but not all graphs.")
        cycle_mask = torch.eq(d.x[choices, 0], torch.ones_like(d.x[choices, 0]))
        _expanded_batch_vector = F.one_hot(d.batch)
        full_path_mask = torch.all((_expanded_batch_vector * d.x[..., 2][..., None]) == _expanded_batch_vector, dim=0)
        stop_mask = d.x[choices, 2]
        reward = full_path_mask * complete_tour_reward + full_path_mask * cycle_mask * complete_cycle_reward \
                 + torch.logical_not(stop_mask) * valid_next_node_reward
        reward = valid_next_step_mask * reward \
                 + torch.logical_not(valid_next_step_mask) * illegal_next_node_reward
        return simulation_running_flag * reward

class SizeIndependentCombinatorialScorer(CombinatorialScorer):
    def batch_reward(self, d: torch_g.data.Batch, choices, simulation_running_flag):
        reward = super().batch_reward(d, choices, simulation_running_flag)
        num_nodes_per_graph = d.num_nodes // d.num_graphs
        return reward / num_nodes_per_graph
