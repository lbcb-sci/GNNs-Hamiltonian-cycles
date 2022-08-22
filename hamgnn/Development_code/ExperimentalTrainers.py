import torch
import torch_geometric as torch_g

from src.Models import HamFinderGNN, HamCycleFinderWithValueFunction
from src.Trainers import SupervisedTrainFollowingHamiltonCycle, REINFORCE_WithLearnableBaseline


class NeighborMaskedSupervisedTrainFollowingHamiltonCycle(SupervisedTrainFollowingHamiltonCycle):
    def _get_next_step_probabilities(self, hamilton_nn: HamFinderGNN, d: torch_g.data.Batch):
        return hamilton_nn.next_step_prob_masked_over_neighbors(d)

    def _get_next_step_logits(self, hamilton_nn: HamFinderGNN, d: torch_g.data.Batch):
        return hamilton_nn.next_step_logits_masked_over_neighbors(d)

    def train_description(self) -> str:
        return "Masking neighbors! " + super().train_description()


class REINFOCE_With_Averaged_simulations(REINFORCE_WithLearnableBaseline):
    def _run_episode(self, nn_hamilton: HamCycleFinderWithValueFunction, optimizer,
                     original_graph: torch_g.data.Data, scorer):
        episode_d = original_graph.clone()
        with torch.no_grad():
            nn_hamilton.init_graph(episode_d)
            nn_hamilton.prepare_for_first_step(episode_d, None)
        nr_steps = 0
        total_reward = 0
        avg_lookahead_list = []

        while True:
            with torch.no_grad():
                a = torch.distributions.Categorical(
                    logits=nn_hamilton.next_step_logits_masked_over_neighbors(episode_d)).sample()
                if episode_d.x[a, 2]:
                    break

            total_reward += scorer.reward(
                episode_d, None if not episode_d.x[..., 1].any() else torch.nonzero(episode_d.x[..., 1]).item(),
                a, nr_steps)
            simulation_d = episode_d.clone()
            if not simulation_d.x[..., 0].any():
                nn_hamilton.prepare_for_next_step(simulation_d, a, a, [a])
            else:
                nn_hamilton.update_state(simulation_d, a)
            state_batches, actions_batches, rewards_batches = \
                self._batch_simulate(simulation_d, nn_hamilton, scorer)
            r = torch.mean(1. * rewards_batches[0])
            loss = self._compute_loss(nn_hamilton, episode_d, a, r)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if not episode_d.x[..., 0].any():
                    nn_hamilton.prepare_for_next_step(episode_d, a, a, [a])
                else:
                    nn_hamilton.update_state(episode_d, a)
            nr_steps += 1
            actions_tensor = torch.stack(actions_batches, dim=-1)
            lookahead = torch.sum(actions_tensor != (-1 * torch.ones_like(actions_tensor)), dim=-1)
            avg_lookahead_list += [torch.mean(1. * lookahead).item() + nr_steps]

        return total_reward, total_reward, sum(avg_lookahead_list) / len(avg_lookahead_list)

    def train_description(self) -> str:
        return f"Trains {self.episodes_per_example} iterations on {self.nr_epochs} examples." \
               f" Averaged over {self.simulation_batch_size}-simulations to reduce variance when updating current state." \
               f" Masking to choose next step only among neighbors."

