import numpy
import torch_geometric as torch_g
import torch
import itertools
import pandas


def _generate_ERmk_model_edge_index_for_small_k(num_nodes, num_edges):
    assert num_edges < num_nodes * (num_nodes - 1) // 4
    original_dtype = numpy.int64
    _generation_overhead = 0.1
    edge_index = numpy.empty([0, 2], dtype=original_dtype)
    while edge_index.shape[0] < num_edges:
        points_to_generate = 2 * (num_edges + int(num_edges * _generation_overhead))
        generated_edges = numpy.random.randint(0, num_nodes, size=[points_to_generate],
                                               dtype=original_dtype)
        symmetrized_edges = numpy.empty(shape=[2*generated_edges.shape[0]], dtype=generated_edges.dtype)
        symmetrized_edges[0::4] = generated_edges[0::2]
        symmetrized_edges[1::4] = generated_edges[1::2]
        symmetrized_edges[2::4] = generated_edges[1::2]
        symmetrized_edges[3::4] = generated_edges[0::2]
        symmetrized_edges = symmetrized_edges.view(dtype=numpy.dtype([("x", original_dtype), ("y", original_dtype)]))
        symmetrized_edges = pandas.unique(symmetrized_edges)
        symmetrized_edges = symmetrized_edges.view(dtype=original_dtype).reshape([symmetrized_edges.shape[0], 2])
        symmetrized_edges = symmetrized_edges[symmetrized_edges[:, 0] != symmetrized_edges[:, 1]]
        new_edges = symmetrized_edges[0::2]
        edge_index = numpy.concatenate([edge_index, new_edges], axis=0)
    edge_index = torch.t(torch.from_numpy(edge_index))
    return edge_index[:, :num_edges]


def generate_ERp_model_edge_index_for_small_k(num_nodes, prob):
    num_edges = numpy.random.binomial(num_nodes * (num_nodes-1) // 2, prob)
    return _generate_ERmk_model_edge_index_for_small_k(num_nodes, num_edges)


class NoisyCycleBatchGenerator:
    def __init__(self, num_nodes, expected_noise_edge_for_node, batch_size=1):
        self.num_nodes = num_nodes
        self.expected_noise_edge_for_node = expected_noise_edge_for_node
        self.batch_size = batch_size

    def _generate_noisy_cycle(self):
        d = torch_g.data.Data()
        ER_edge_index = generate_ERp_model_edge_index_for_small_k(self.num_nodes,
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
        d.edge_index = generate_ERp_model_edge_index_for_small_k(d.num_nodes, self.p)
        return d

    def output_details(self):
        return f"ER({self.num_nodes}, {self.p})"

    def __iter__(self):
        return (self._erdos_renyi_generator() for i in itertools.count())
