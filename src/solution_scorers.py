import torch
import torch.nn.functional as F
import torch_geometric as torch_g
from abc import ABC

import src.Models as Models

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
        valid_next_node_reward = self.valid_next_node_reward
        illegal_next_node_reward = self.illegal_next_node_multiplier * d.num_nodes
        complete_cycle_reward = self.complete_cycle_multiplier * d.num_nodes
        complete_tour_reward = self.complete_tour_multiplier * d.num_nodes

        neighbor_indices = Models.HamFinderGNN._neighbor_indices(d)
        valid_next_step_mask = torch.isin(choices, neighbor_indices)
        cycle_mask = torch.equal(d.x[choices, 0], torch.ones_like(d.x[choices, 0]))
        _expanded_batch_vector = F.one_hot(d.batch)
        full_path_mask = torch.all((_expanded_batch_vector * d.x[..., 2][..., None]) == _expanded_batch_vector, dim=0)
        stop_mask = d.x[choices, 2]
        reward = full_path_mask * complete_tour_reward + full_path_mask * cycle_mask * complete_cycle_reward \
                 + torch.logical_not(stop_mask) * valid_next_node_reward
        reward = valid_next_step_mask * reward \
                 + torch.logical_not(valid_next_step_mask) * illegal_next_node_reward
        return simulation_running_flag * reward

        # TODO if statement unnecesarry
        if simulation_running_flag is None:
            simulation_running_flag = torch.ones_like(choices)
        x = d.x.reshape([d.num_graphs, -1, 3])

        choices_mask = torch.zeros_like(d.x[..., 2]).scatter_(0, choices, 1).reshape([d.num_graphs, -1])
        valid_next_steps_mask = Models.HamFinderGNN._neighbor_indices(d).reshape([d.num_graphs, -1])
        illegal_next_step_mask = torch.logical_not(torch.minimum(valid_next_steps_mask, choices_mask).any())

        cycle_mask = torch.minimum(x[..., 0], choices_mask).any(dim=-1)
        full_path_mask = x[..., 2].all(-1)
        stop_mask = torch.minimum(x[..., 2], choices_mask).any(-1)
        reward = full_path_mask * complete_tour_reward + full_path_mask * cycle_mask * complete_cycle_reward \
                 + torch.logical_not(stop_mask) * valid_next_node_reward
        reward = torch.logical_not(illegal_next_step_mask) * reward \
                 + illegal_next_step_mask * illegal_next_node_reward
        return simulation_running_flag * reward
