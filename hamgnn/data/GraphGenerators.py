import numpy
import torch_geometric as torch_g
import torch
import itertools
from typing import List, Iterator
from abc import ABC, abstractmethod

from hamgnn.data.GraphDataset import GraphExample


class GraphGenerator(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[GraphExample]:
        pass


def _generate_ERmk_for_small_k(num_nodes, num_edges):
    _edge_fraction = 2 * num_edges / (num_nodes * (num_nodes - 1))
    if _edge_fraction > 1/2:
        print(f"Using inefficient method (ment for sparse graph with edge fraction << 1) for Erdos-Renyi graph with edge fraction {_edge_fraction}")
    original_dtype = numpy.int64
    _generation_overhead_fraction = 0.1

    edges = set()
    while len(edges) < num_edges:
        points_to_generate = num_edges + int(num_edges * _generation_overhead_fraction)
        generated_edges = numpy.random.randint(0, num_nodes, size=[points_to_generate, 2],
                                               dtype=original_dtype)
        generated_edges = generated_edges[generated_edges[:, 0] < generated_edges[:, 1]]
        new_edges = [(e[0], e[1]) for e in generated_edges]
        edges.update(new_edges[:num_edges - len(edges)])

    edge_index = numpy.array([[e[0], e[1]] for e in edges], dtype=original_dtype)
    edge_index = numpy.concatenate([edge_index, numpy.flip(edge_index, axis=-1)], axis=0)
    edge_index = torch.t(torch.from_numpy(edge_index))
    return edge_index


def generate_ERp_for_small_p(num_nodes, prob):
    num_edges = numpy.random.binomial(num_nodes * (num_nodes-1) // 2, prob)
    return _generate_ERmk_for_small_k(num_nodes, num_edges)


class NoisyCycleGenerator(GraphGenerator):
    def __init__(self, num_nodes, expected_noise_edges_per_node):
        self.num_nodes = num_nodes
        self.expected_noise_edges_per_node = expected_noise_edges_per_node

    def _generate_noisy_cycle(self):
        d = torch_g.data.Data()
        ER_edge_index = generate_ERp_for_small_p(self.num_nodes,
                                                        self.expected_noise_edges_per_node / (self.num_nodes-1))
        artificial_cycle = torch.randperm(self.num_nodes)
        artificial_edges = torch.stack([artificial_cycle, artificial_cycle.roll(-1, 0)], dim=0)
        artificial_edges = torch.cat([artificial_edges, artificial_edges.flip(dims=(-2,))], dim=-1)
        artificial_cycle = torch.cat([artificial_cycle, artificial_cycle[0].unsqueeze(0)], dim=0)
        d.num_nodes = self.num_nodes
        d.edge_index = torch.cat([ER_edge_index, artificial_edges], dim=-1)
        d.edge_index = d.edge_index.unique(dim=-1)
        choice_distribution = None
        return GraphExample(d, artificial_cycle, choice_distribution)

    def output_details(self):
        return f"A cycle of with expected {self.expected_noise_edges_per_node} noise edge per node"

    def __iter__(self):
        return (self._generate_noisy_cycle() for _ in itertools.count())


class ErdosRenyiGenerator:
    def __init__(self, num_nodes, hamilton_existence_probability):
        assert num_nodes > 2
        self.num_nodes = num_nodes
        self.hamilton_existence_probability = hamilton_existence_probability

        # see Komlos, Szemeredi - Limit distribution for the existence of hamiltonian cycles in a random graph
        c = -numpy.log(-numpy.log(self.hamilton_existence_probability)) / 2
        self.p = numpy.log(num_nodes) / (num_nodes - 1) \
                 + numpy.log(numpy.log(num_nodes)) / (num_nodes - 1) + 2 * c / (num_nodes - 1)

    @staticmethod
    def create_from_edge_probability(num_nodes, edge_existence_probability):
        ER_generator = ErdosRenyiExamplesGenerator(num_nodes, 1)
        ER_generator.hamilton_existence_probability = None
        ER_generator.p = edge_existence_probability

    def _erdos_renyi_generator(self):
        d = torch_g.data.Data()
        d.num_nodes = self.num_nodes
        d.edge_index = generate_ERp_for_small_p(d.num_nodes, self.p)
        return d

    def output_details(self):
        return f"ER({self.num_nodes}, {self.p})"

    def __iter__(self):
        return (self._erdos_renyi_generator() for _ in itertools.count())


class ErdosRenyiExamplesGenerator:
    def __init__(self, num_nodes, hamilton_existence_probability):
        self.raw_generator = ErdosRenyiGenerator(num_nodes, hamilton_existence_probability)

    def output_details(self):
        return f"{self.raw_generator.output_details()} packed as {GraphExample.__name__}"

    def __iter__(self):
        return (GraphExample(graph, None, None) for graph in self.raw_generator)
