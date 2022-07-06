import numpy
from tqdm import tqdm
from collections import defaultdict

from src.data.InMemoryDataset import ErdosRenyiInMemoryDataset
from src.data.GraphGenerators import generate_ERp_model_edge_index_for_small_k
from src.data.InMemoryDataset import ErdosRenyiGraphExample
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


if __name__ == "__main__":
    nr_graphs = 10000
    prob = 0.2
    graphs = []
    for num_nodes in [25, 50, 100, 150]:
        edge_indexes = [generate_ERp_model_edge_index_for_small_k(num_nodes, prob) for _ in range(nr_graphs)]
        graphs.extend([Data(edge_index=ei, num_nodes=num_nodes) for ei in edge_indexes])
    examples = [ErdosRenyiGraphExample(g, prob, None) for g in graphs]
    test_ER_dataset(examples)
    print("Sucessfully tested on generating graphs.")

    from pathlib import Path
    dataset = ErdosRenyiInMemoryDataset([Path("DATA").resolve()])
    test_ER_dataset(dataset.data_list)
    print("Sucessfully tested on stored graphs.")
