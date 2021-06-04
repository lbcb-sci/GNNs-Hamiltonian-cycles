import numpy
import torch_geometric as torch_g
import torch
import itertools


class NoisyCycleBatchGenerator:
    def __init__(self, num_nodes, expected_noise_edge_for_node, batch_size=1):
        self.num_nodes = num_nodes
        self.expected_noise_edge_for_node = expected_noise_edge_for_node
        self.batch_size = batch_size

    def _generate_noisy_cycle(self):
        d = torch_g.data.Data()
        ER_edge_index = torch_g.utils.erdos_renyi_graph(self.num_nodes,
                                                        self.expected_noise_edge_for_node / self.num_nodes)
        artificial_cycle = torch.randperm(self.num_nodes)
        artificial_edges = torch.stack([artificial_cycle, artificial_cycle.roll(-1, 0)], dim=0)
        artificial_edges = torch.cat([artificial_edges, artificial_edges.flip(dims=(-2,))], dim=-1)
        artificial_cycle = torch.cat([artificial_cycle, artificial_cycle[0].unsqueeze(0)], dim=0)
        d.num_nodes = self.num_nodes
        d.edge_index = torch.cat([ER_edge_index, artificial_edges], dim=-1)
        return d, artificial_cycle

    def _generate_batch(self):
        graphs = []
        cycles = []
        for _ in range(self.batch_size):
            g, c = self._generate_noisy_cycle()
            graphs += [g]
            cycles += [c]
        batch = torch_g.data.Batch.from_data_list(graphs)
        cycles = torch.stack(cycles)
        cycles = cycles + self.num_nodes * torch.arange(0, self.batch_size)[..., None]
        return batch, cycles, None

    def output_details(self):
        return f"Batch of {self.batch_size} {self.num_nodes}-cycles with expected {self.expected_noise_edge_for_node}" \
               f" noise edge per node"

    def __iter__(self):
        return (self._generate_batch() for _ in itertools.count())


class ErdosRenyiGenerator:
    def __init__(self, num_nodes, hamilton_existence_probability):
        assert num_nodes > 2
        self.num_nodes = num_nodes
        self.hamilton_existence_probability = hamilton_existence_probability

        # see Komlos, Szemeredi - Limit distribution for the existence of hamiltonian cycles in a random graph
        c = -numpy.log(-numpy.log(self.hamilton_existence_probability)) / 2
        self.p = numpy.log(num_nodes) / (num_nodes - 1) \
                 + numpy.log(numpy.log(num_nodes)) / (num_nodes - 1) + 2 * c / (num_nodes - 1)

    def _erdos_renyi_generator(self):
        d = torch_g.data.Data()
        d.num_nodes = self.num_nodes
        d.edge_index = torch_g.utils.erdos_renyi_graph(d.num_nodes, self.p)
        return d

    def output_details(self):
        return f"ER({self.num_nodes}, {self.p})"

    def __iter__(self):
        return (self._erdos_renyi_generator() for i in itertools.count())
