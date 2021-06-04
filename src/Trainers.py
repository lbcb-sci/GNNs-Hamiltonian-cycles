import sys
import time
import os
from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch_geometric as torch_g
from matplotlib import pyplot as plt

from src.Models import HamiltonianCycleFinder, DEVICE, HamiltonCycleFinderWithValueFunction


class TrainAlgorithm(ABC):
    def __init__(self, verbose=1):
        self._print_stream = open(os.devnull, 'w') if verbose == 0 else sys.stdout

    def print_message(self, message):
        print(message, file=self._print_stream)

    @abstractmethod
    def _non_measured_train(self, generator, nn_hamilton, optimizer) -> Dict[str, str]: pass

    def train(self, generator, nn_hamilton, optimizer) -> Dict[str, str]:
        start_time = time.process_time()
        history = self._non_measured_train(generator, nn_hamilton, optimizer)
        history["train_time"] = time.process_time() - start_time
        return history

    @abstractmethod
    def train_description(self) -> str: pass

    @abstractmethod
    def loss_description(self) -> str: pass


class SupervisedTrainFollowingHamiltonCycle(TrainAlgorithm):
    def __init__(self, nr_epochs, iterations_per_epoch, loss_type="entropy", verbose=1, batch_size=1,
                 learning_rate_decrease_condition_window=None, learning_rate_decrease_factor=0.5,
                 terminating_learning_rate=None):
        super(SupervisedTrainFollowingHamiltonCycle, self).__init__(verbose)
        self.nr_epochs = nr_epochs
        self.iterations_per_epoch = iterations_per_epoch
        self.loss_type = loss_type
        self.batch_size = batch_size
        self.learning_rate_decrease_condition_window = learning_rate_decrease_condition_window
        self.learning_rate_decrease_factor = learning_rate_decrease_factor
        self.terminating_learning_rate = terminating_learning_rate

    def _get_next_step_logits(self, hamilton_nn: HamiltonianCycleFinder, d: torch_g.data.Batch):
        return hamilton_nn.next_step_logits(d)

    def _get_next_step_probabilities(self, hamilton_nn: HamiltonianCycleFinder, d: torch_g.data.Batch):
        return hamilton_nn.next_step_prob(d)

    def _compute_loss(self, d, teacher_paths, teacher_distributions, hamilton_nn: HamiltonianCycleFinder):
        assert teacher_paths is not None
        d = d.to(hamilton_nn.get_device())
        teacher_paths = teacher_paths.to(hamilton_nn.get_device())
        hamilton_nn.init_graph(d)
        hamilton_nn.prepare_for_first_step(d, None)
        hamilton_nn.next_step_logits(d)
        hamilton_nn.prepare_for_first_step(d, teacher_paths[..., 0])

        loss = torch.zeros(1, device=DEVICE)
        for step in range(1, d.num_nodes // d.num_graphs + 1):
            if step > 1:
                hamilton_nn.update_state(d, teacher_paths[..., step - 1])
            targets = teacher_paths[..., step]
            targets = targets - (d.num_nodes // d.num_graphs) * d.batch[targets]
            if self.loss_type == "mse":
                p = self._get_next_step_probabilities(hamilton_nn, d)
                mse_loss = torch.nn.MSELoss()
                loss += mse_loss(p, torch.zeros_like(p).index_fill(0, teacher_paths[..., step], 1.))
            elif self.loss_type == "entropy":
                logits = self._get_next_step_logits(hamilton_nn, d).reshape([self.batch_size, -1])
                entropy_loss = torch.nn.CrossEntropyLoss()
                loss += entropy_loss(logits, targets)
            else:
                self.print_message("{} not a valid loss type".format(self.loss_type))
                loss = None
        return loss

    def _non_measured_train(self, graph_generator, hamilton_nn: HamiltonianCycleFinder, optimizer):
        history = {"epoch": [], "epoch_max": [], "epoch_avg": [], "avg_l2_parameters": []}
        graph_it = iter(graph_generator)
        learn_rate_decay_delay = self.learning_rate_decrease_condition_window

        for epoch in range(self.nr_epochs):
            _start_time = time.process_time()
            total_loss = torch.zeros(1, device=hamilton_nn.get_device())
            max_loss = None

            for iteration in range(self.iterations_per_epoch):
                d, teacher_paths, teacher_distributions = next(graph_it)
                loss = self._compute_loss(d, teacher_paths, teacher_distributions, hamilton_nn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if max_loss is None:
                    max_loss = loss.detach()
                else:
                    max_loss = torch.maximum(max_loss, loss.detach())
                total_loss += loss.detach()

            avg_loss = total_loss.item() / self.iterations_per_epoch
            total_l2_norm = sum([torch.sum(torch.square(p.detach())) for p in hamilton_nn.parameters()])
            if self.learning_rate_decrease_condition_window and learn_rate_decay_delay == 0:
                recent_losses = history["epoch_avg"][-self.learning_rate_decrease_condition_window:]
                if avg_loss > sum(recent_losses)/len(recent_losses):
                    approximate_lr = 0
                    learn_rate_decay_delay = self.learning_rate_decrease_condition_window
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] * self.learning_rate_decrease_factor
                        approximate_lr += g['lr']
                    approximate_lr /= len(optimizer.param_groups)
                    print(f"Reducing learning rate by {self.learning_rate_decrease_factor} to {approximate_lr}.")
            if self.learning_rate_decrease_condition_window:
                learn_rate_decay_delay = learn_rate_decay_delay - 1 if learn_rate_decay_delay > 0 else 0

            history["epoch"] += [epoch]
            history["epoch_max"] += [max_loss.item()]
            history["epoch_avg"] += [avg_loss]
            history["avg_l2_parameters"] += [
                total_l2_norm / self.iterations_per_epoch / len(list(hamilton_nn.parameters()))]
            self.print_message(
                "Epoch {}, max_loss: {:.2f}, avg_loss {:.2f}:  ({:.2f} s)".format(
                    epoch + 1, max_loss.item(), avg_loss, time.process_time() - _start_time))
            avg_learn_rate = sum([g['lr'] for g in optimizer.param_groups]) \
                / len(optimizer.param_groups)
            if self.terminating_learning_rate is not None and avg_learn_rate  < self.terminating_learning_rate:
                print(f"Stopping learning loop, current learn rate {avg_learn_rate} is below terminating rate"
                      f" {self.terminating_learning_rate}")
                break
        return history

    def loss_description(self) -> str:
        return f"Teacher forcing supervised loss. Minimizes {self.loss_type} between teacher and student next node" \
               f" distributions. Teacher follows a pregenerated hamilton cycle (no preferences) and initial student" \
               f" step is not scored."

    def train_description(self) -> str:
        return f"Train for {self.nr_epochs}x{self.iterations_per_epoch} batches of {self.batch_size} examples"


class ReinforcementScorer(ABC):
    def batch_reward(self, d: torch_g.data.Batch, choices, simulation_running_flag=None) -> torch.Tensor:
        pass


class CombinatorialScorer(ReinforcementScorer):
    def __init__(self, valid_next_node_reward, illegal_next_node_multiplier, complete_cycle_multiplier,
                 complete_tour_multiplier):
        self.valid_next_node_reward = valid_next_node_reward
        self.illegal_next_node_multiplier = illegal_next_node_multiplier
        self.complete_cycle_multiplier = complete_cycle_multiplier
        self.complete_tour_multiplier = complete_tour_multiplier

    def batch_reward(self, d: torch_g.data.Batch, choices, simulation_running_flag=None):
        valid_next_node_reward = self.valid_next_node_reward
        illegal_next_node_reward = self.illegal_next_node_multiplier * d.num_nodes
        complete_cycle_reward = self.complete_cycle_multiplier * d.num_nodes
        complete_tour_reward = self.complete_tour_multiplier * d.num_nodes

        if simulation_running_flag is None:
            simulation_running_flag = torch.ones_like(choices)
        x = d.x.reshape([d.num_graphs, -1, 3])

        choices_mask = torch.zeros_like(d.x[..., 2]).scatter_(0, choices, 1).reshape([d.num_graphs, -1])
        valid_next_steps_mask = HamiltonianCycleFinder._neighbor_mask(d).reshape([d.num_graphs, -1])
        illegal_next_step_mask = torch.logical_not(torch.minimum(valid_next_steps_mask, choices_mask).any())

        cycle_mask = torch.minimum(x[..., 0], choices_mask).any(dim=-1)
        full_path_mask = x[..., 2].all(-1)
        stop_mask = torch.minimum(x[..., 2], choices_mask).any(-1)
        reward = full_path_mask * complete_tour_reward + full_path_mask * cycle_mask * complete_cycle_reward \
                 + torch.logical_not(stop_mask) * valid_next_node_reward
        reward = torch.logical_not(illegal_next_step_mask) * reward \
                 + illegal_next_step_mask * illegal_next_node_reward
        return simulation_running_flag * reward


class REINFORCE_WithLearnableBaseline(TrainAlgorithm):
    def __init__(self, nr_epochs, iterations_in_epoch, episodes_per_example=1, scorer=CombinatorialScorer(1, -1, 2, 1),
                 l2_regularization_weight=0.01, value_function_weight=1,
                 simulation_batch_size=8, verbose=1, max_simulation_steps=None):
        super(REINFORCE_WithLearnableBaseline, self).__init__(verbose)
        self.nr_epochs = nr_epochs
        self.iterations_in_epoch = iterations_in_epoch
        self.episodes_per_example = episodes_per_example
        self.scorer = scorer
        self.l2_regularization_weight = l2_regularization_weight
        self.simulation_batch_size = simulation_batch_size
        self.value_function_weight = value_function_weight
        self.max_simulation_steps = max_simulation_steps

    def create_simulation_batch(self, d):
        return torch_g.data.Batch.from_data_list(self.simulation_batch_size * [d])

    def _batch_simulate(self, d: torch_g.data.Data, nn_hamilton: HamiltonianCycleFinder, scorer):
        max_simulation_steps = self.max_simulation_steps
        if max_simulation_steps is None:
            max_simulation_steps = d.num_nodes
        with torch.no_grad():
            d = self.create_simulation_batch(d)
            simulation_running_flag = torch.ones([d.num_graphs], dtype=torch.bool, device=d.x.device)
            states, actions, rewards = [], [], []
            step_no = 0
            while simulation_running_flag.any() and step_no < max_simulation_steps + 1:
                step_no += 1
                logits = nn_hamilton.next_step_logits_masked_over_neighbors(d).reshape([self.simulation_batch_size, -1])
                q = torch.distributions.Categorical(logits=logits)
                choices = q.sample() + d.num_nodes // d.num_graphs * torch.arange(d.num_graphs)
                states += [d.clone()]
                actions += [simulation_running_flag * choices + (-1) * torch.logical_not(simulation_running_flag)]
                rewards += [scorer.batch_reward(d, choices, simulation_running_flag)]
                choices_mask = torch.zeros_like(d.x[..., 2]).scatter_(0, choices, 1).reshape([d.num_graphs, -1])
                simulation_running_flag = torch.minimum(
                    simulation_running_flag, torch.logical_not(torch.minimum(
                        choices_mask, d.x[..., 2].reshape([d.num_graphs, -1])).any(-1)))
                if not d.x[..., 0].any():
                    nn_hamilton.prepare_for_first_step(d, choices)
                else:
                    nn_hamilton.update_state(d, choices)
            for i in range(len(rewards) - 2, -1, -1):
                rewards[i] += rewards[i + 1]
            return states, actions, rewards

    def _compute_loss(self, nn_hamilton, state, action, reward):
        logits, value_estimate = nn_hamilton.next_step_logits_and_value_function(state)
        q = torch.distributions.Categorical(logits=logits)
        l2_params = torch.sum(torch.stack([torch.sum(torch.square(p)) for p in nn_hamilton.parameters()]))
        value_estimate_loss = torch.square(reward - value_estimate)
        REINFORCE_loss = -(reward - value_estimate).detach() * q.log_prob(action)
        total_loss = REINFORCE_loss + self.l2_regularization_weight * l2_params + self.value_function_weight * value_estimate_loss
        return total_loss

    def _run_episode(self, nn_hamilton: HamiltonCycleFinderWithValueFunction,
                     optimizer, original_graph: torch_g.data.Data, scorer):
        d = original_graph.clone()
        nn_hamilton.init_graph(d)
        nn_hamilton.prepare_for_first_step(d, None)
        state_batches, actions_batches, rewards_batches = \
            self._batch_simulate(d, nn_hamilton, scorer)

        states = [s for b in state_batches for s in b.to_data_list()]
        actions = [a if a > -1 else -1 for b in actions_batches for a in
                   (b - d.num_nodes * torch.arange(b.shape[0])).split(1)]
        rewards = [r for b in rewards_batches for r in b.split(1)]

        for (s, a, r) in zip(states, actions, rewards):
            if a <= -1:
                continue
            loss = self._compute_loss(nn_hamilton, s, a , r)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_reward = torch.mean(1. * rewards_batches[0]).item()
        actions_tensor = torch.stack(actions_batches, -1)
        lookahead = torch.sum(actions_tensor != (-1 * torch.ones_like(actions_tensor)), dim=-1)
        avg_lookahead = torch.mean(1. * lookahead).item()
        min_reward = torch.min(rewards_batches[0]).item()
        return avg_reward, min_reward, avg_lookahead

    def _non_measured_train(self, generator, nn_hamilton: HamiltonCycleFinderWithValueFunction, optimizer):
        history = {"epoch": [], "epoch_max": [], "epoch_avg": [], "l2_parameters": []}
        epoch = 0
        it = iter(generator)
        while epoch < self.nr_epochs:
            min_reward = None
            total_reward = 0
            total_lookahead = 0
            _iteration_start_time = time.process_time()
            epoch += 1
            for _ in range(self.iterations_in_epoch):
                original_graph = next(it)
                original_graph.to(nn_hamilton.get_device())
                for episode in range(self.episodes_per_example):
                    episode_avg_reward, episode_min_reward, episode_avg_lookahead \
                        = self._run_episode(nn_hamilton, optimizer, original_graph, self.scorer)
                    total_reward += episode_avg_reward
                    total_lookahead += episode_avg_lookahead
                    if min_reward is None:
                        min_reward = episode_min_reward
                    else:
                        min_reward = min(min_reward, episode_min_reward)

            avg_loss = total_reward / self.episodes_per_example / self.iterations_in_epoch
            history["epoch"] += [epoch]
            history["epoch_max"] += [-1 * min_reward]
            history["epoch_avg"] += [-1 * avg_loss]
            parmeters = list(nn_hamilton.parameters())
            avg_lookahead = total_lookahead / self.episodes_per_example / self.iterations_in_epoch
            history["l2_parameters"] += [
                torch.sum(torch.stack([torch.sum(torch.square(p)) / len(parmeters) for p in parmeters])).item()]
            self.print_message(
                "Epoch {}, min reward: {:.2f}, avg reward {:.2f}, avg lookahead {:.2f} steps:  ({:.2f} s)"
                               .format(epoch, min_reward, avg_loss, avg_lookahead,
                                       time.process_time() - _iteration_start_time))
        return history

    def loss_description(self) -> str:
        return f"Reinforcement loss with learnable value function as baseline." \
               f"loss(\\theta) = -(G_t - V(S_t, \\theta)) ln \\pi(A_t, \\theta | S_t) +" \
               f" {self.value_function_weight}*(V(S_t, \\theta) - G_t)^2" \
               f" + {self.l2_regularization_weight}*l_2(\\theta)." \
               f" G_t is the cumulative reward of a simulation and V(S_t, \\theta) is the state value function"

    def train_description(self) -> str:
        return f"Trains for {self.iterations_in_epoch} epochs consisting of {self.iterations_in_epoch} independent" \
               f" examples. {self.episodes_per_example} batches (size {self.simulation_batch_size}) simulations" \
               f" per example. After the batch of simulation finishes, all visited states are updated." \
               f" Masking to choose next step only among neighbors."


def plot_history(history, out_file=None, loss_name="cross entropy"):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(history["epoch"], history["epoch_max"], color="red", label="max")
    ax.plot(history["epoch"], history["epoch_avg"], color="blue", label="avg")
    ax.set_xlabel("epochs")
    ax.set_ylabel(loss_name)
    ax.legend(loc="upper right")
    if out_file is None:
        fig.show()
    else:
        fig.savefig(out_file, bbox_inches="tight")
