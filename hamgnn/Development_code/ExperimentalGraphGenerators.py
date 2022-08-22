import torch
import torch_geometric as torch_g
import itertools

from src.GraphGenerators import ErdosRenyiGenerator


class NoisyCycleGenerator:
    def __init__(self, num_nodes, nr_cycles, expected_noise_edges_for_node):
        assert self.num_nodes > 1
        self.num_nodes = num_nodes
        self.nr_cycles = nr_cycles
        self.expected_noise_edges_for_node = expected_noise_edges_for_node

    def _generate_noisy_cycle(self):
        d = torch_g.data.Data()
        d.num_nodes = self.num_nodes
        ER_edge_index = torch_g.utils.erdos_renyi_graph(self.num_nodes,
                                                        self.expected_noise_edges_for_node / (self.num_nodes - 1))
        artificial_cycles = [torch.randperm(self.num_nodes) for _ in range(self.nr_cycles)]
        artificial_edges = [torch.stack([c, torch.roll(c, 1)], dim=0) for c in artificial_cycles]
        d.edge_index = torch.cat([ER_edge_index] + artificial_edges, dim=1)
        d.edge_index = torch_g.utils.to_undirected(d.edge_index, self.num_nodes)
        artificial_cycles = torch.stack(artificial_cycles
                                        + [torch.flip(c, dims=[0]) for c in artificial_cycles]).unique(False, dim=0)
        artificial_cycles = torch.cat([artificial_cycles, artificial_cycles[..., 0][..., None]], dim=1)

        tour, distributions = _compute_teacher_route(d, torch.stack(list(artificial_cycles)))
        return torch_g.data.Batch.from_data_list([d]), torch.unsqueeze(tour, 0), torch.unsqueeze(distributions, 0)

    def __iter__(self):
        return (self._generate_noisy_cycle() for _ in itertools.count())

    def output_details(self):
        start = f"{self.nr_cycles} independent " if self.nr_cycles > 1 else ""
        return f"{start}{self.num_nodes}-cycles with expected {self.expected_noise_edges_for_node}" \
               f" extra edges per node"


def _compute_teacher_route(d: torch_g.data.Data, hamilton_cycles):
    teacher_path = hamilton_cycles[0]
    distributions = torch.zeros([d.num_nodes, d.num_nodes])
    for step in range(0, len(hamilton_cycles[0]) - 1):
        valid_cycles = [hc for hc in hamilton_cycles if torch.eq(hc[:step + 1], teacher_path[:step + 1]).all()]
        for c in valid_cycles:
            distributions[step, c[step + 1]] += 1
        distributions[step] = distributions[step] / len(valid_cycles)
    return teacher_path, distributions


class ErdosRenyiBatchGenerator(ErdosRenyiGenerator):
    def __init__(self, num_nodes, hamilton_existence_probability, batch_size):
        super(ErdosRenyiBatchGenerator, self).__init__(num_nodes, hamilton_existence_probability)
        self.batch_size = batch_size

    def output_details(self):
        return f"Batches of {self.batch_size} ER({self.num_nodes}, {self.p})"

    def _erdos_renyi_batch_generator(self):
        return torch_g.data.Batch.from_data_list([self._erdos_renyi_generator() for _ in range(self.batch_size)])

    def __iter__(self):
        return (self._erdos_renyi_batch_generator() for _ in itertools.count())
