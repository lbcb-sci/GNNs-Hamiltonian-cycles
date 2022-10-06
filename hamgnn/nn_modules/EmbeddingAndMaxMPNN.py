import torch
import torch_geometric as torch_g
import torchinfo

from hamgnn.HamiltonSolver import DataUtils
from hamgnn.nn_modules.nn_modules import ResidualMultilayerMPNN
from hamgnn.data.GraphDataset import BatchedSimulationStates, GraphBatchExample, GraphExample, SimulationState
import hamgnn.solution_scorers as scorers
import hamgnn.nn_modules.hamilton_gnn_utils as gnn_utils


class EmbeddingAndMaxMPNN(gnn_utils.HamCycleFinderWithValueFunction):
    def _construct_embedding(self):
        embedding = ResidualMultilayerMPNN(self.hidden_dim, self.hidden_dim, 1, nr_layers=self.embedding_depth)
        embedding_out_projection = torch.nn.Linear(self.hidden_dim, self.hidden_dim - self.in_dim)
        return embedding, embedding_out_projection

    def _construct_processor(self):
        processor = ResidualMultilayerMPNN(self.hidden_dim, self.hidden_dim, 1, nr_layers=self.processor_depth)
        processor_out_projection = torch.nn.Linear(self.hidden_dim, self.out_dim)
        return processor, processor_out_projection

    def __init__(self, in_dim=3, out_dim=2, hidden_dim=32, embedding_depth=5, processor_depth=5,
                 value_function_weight=1, l2_regularization_weight=0.01, nr_simultaneous_simulations=8,
                 loss_type="mse", graph_updater_class=gnn_utils.WalkUpdater, solution_scorer_class=scorers.SizeIndependentCombinatorialScorer, **kwargs):
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
        assert hidden_dim > self.in_dim
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

    def validation_step(self, simulations_dict, batch_ids, dataloader_idx=0):
        with torch.no_grad():
            loss = self.training_step(simulations_dict)
            self.log("validation/loss", loss)

            graphs = [graph for graph in BatchedSimulationStates.from_lightning_dict(simulations_dict).batch_graph.to_data_list() if len(DataUtils._starting_indices(graph)) == 0]
            if len(graphs) > 0:
                self._validation_nr_different_graphs_seen += len(graphs)
                batch_example = GraphBatchExample.from_graph_examples_list([GraphExample(graph, torch.zeros(25, dtype=graph.edge_index.dtype, device=graph.edge_index.device)) for graph in graphs])
                self._compute_and_update_accuracy_metrics(batch_example.to_lightning_dict(), "val_rl")
        return loss


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
