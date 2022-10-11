import numpy
from tqdm import tqdm
from collections import defaultdict

from hamgnn.data.InMemoryDataset import ErdosRenyiInMemoryDataset
from hamgnn.data.GraphGenerators import generate_ERp_for_small_p, NoisyCycleGenerator
from hamgnn.data.InMemoryDataset import ErdosRenyiGraphExample
from torch_geometric.data import Data


def test_ER_dataset(all_graph_examples):
    group_to_graphs_map = defaultdict(list)
    for graph_example in tqdm(all_graph_examples):
        group_to_graphs_map[(graph_example.graph.num_nodes, graph_example.edge_inclusion_probability)].append(graph_example)

    for (size, edge_inclusion_probability), graph_examples in tqdm(group_to_graphs_map.items()):
        edge_index = numpy.concatenate([example.graph.edge_index.numpy() for example in tqdm(graph_examples, leave=False)], axis=1)
        (unique, counts) = numpy.unique(edge_index, axis=-1, return_counts=True)
        (_reversed_unique, _reversed_counts) = numpy.unique(numpy.flip(edge_index, axis=0), axis=-1, return_counts=True)

        assert numpy.all(unique == _reversed_unique), "Not all graphs are symmetric"
        assert numpy.all(counts == _reversed_counts), f"There are {numpy.sum(counts - _reversed_counts)} symmetric edges missing"

        counts = counts[unique[0] < unique[1]]
        nr_graphs = len(graph_examples)
        _std = numpy.sqrt(edge_inclusion_probability * (1 - edge_inclusion_probability) * nr_graphs)
        counts_normalized = (counts - edge_inclusion_probability * nr_graphs) / _std
        perc_suspicious = numpy.mean(counts_normalized > 2)
        assert perc_suspicious < 0.035, f"{perc_suspicious * 100}% of edges behave unexpectedly"


def _count_neighbors(graph_example):
    unique, counts = numpy.unique(graph_example.graph.edge_index[0, :].numpy(), return_counts=True)
    return unique, counts


def test_generated_ER():
    nr_graphs = 10000
    prob = 0.2
    graphs = []
    for num_nodes in [25, 50, 100, 150]:
        edge_indexes = [generate_ERp_for_small_p(num_nodes, prob) for _ in range(nr_graphs)]
        graphs.extend([Data(edge_index=ei, num_nodes=num_nodes) for ei in edge_indexes])
    examples = [ErdosRenyiGraphExample(g, prob, None) for g in graphs]
    test_ER_dataset(examples)
    print("Sucessfully tested on generating graphs.")


def test_stored_ER_dataset():
    from pathlib import Path
    dataset = ErdosRenyiInMemoryDataset([Path("DATA").resolve()])
    test_ER_dataset(dataset.data_list)
    print("Sucessfully tested on stored graphs.")


def test_noisy_cycle():
    params_nr_nodes = [20, 25, 30, 50]
    params_edges_per_node = [3, 3, 3, 3]
    params_allowed_deviation = [0.1, 0.1, 0.1, 0.1]
    for nr_nodes, nr_edges_per_node, allowed_deviation in zip(params_nr_nodes, params_edges_per_node, params_allowed_deviation):
        empirical_counts = []
        generator = iter(NoisyCycleGenerator(nr_nodes, nr_edges_per_node))
        for _ in range(10):
            g = next(generator)
            nodes, counts = _count_neighbors(g)
            empirical_counts.append(numpy.sum(counts) / nr_nodes)
        deviation = sum(empirical_counts) / len(empirical_counts) - nr_edges_per_node
        assert (-allowed_deviation < deviation) and (deviation < 2 + allowed_deviation) # 2 additional edge per node allowed because of inserted cycle


if __name__ == "__main__":
    test_noisy_cycle()
    test_stored_ER_dataset()
    test_generated_ER()
