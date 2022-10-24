from pathlib import Path
import itertools
import torch_geometric
import networkx
from matplotlib import pyplot as plt
from copy import deepcopy

from hamgnn.data.GraphGenerators import ErdosRenyiGenerator
from hamgnn.heuristics import *
import hamgnn.Evaluation as eval


def _display_HybridHam():
    num_nodes = 10
    generator = ErdosRenyiGenerator(num_nodes, 0.4)

    for d in itertools.islice(generator, 10):
        path = least_degree_first_heuristics(num_nodes, d.edge_index, False)
        path_improved = least_degree_first_heuristics(num_nodes, d.edge_index, True)
        HybridHam_path = HybridHam().solve(d)
        print(len(HybridHam_path), len(path_improved))
        if len(HybridHam_path) == num_nodes + 1:
            nx_graph = torch_geometric.utils.to_networkx(d)
            pos = networkx.spring_layout(nx_graph)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            for ax in (ax1, ax2):
                nx.draw(nx_graph, pos, with_labels=True, ax=ax, arrows=False)

            HybridHam_path_edges = [(HybridHam_path[i], HybridHam_path[i + 1]) for i in range(len(HybridHam_path) - 1)]
            nx_graph.remove_edges_from(deepcopy(nx_graph.edges))
            nx_graph.add_edges_from(HybridHam_path_edges)
            nx.draw_networkx_edges(nx_graph, pos, edge_color="red", arrows=False, ax=ax2)
            plt.show()


if __name__ == '__main__':
    # path = Path(__file__).parent.parent.parent / "HCP_benchmarks/graph2.hcp"
    # num_nodes, edge_index = load_graph_from_hcp_file(path)
    # graph = torch_geometric.data.Data(num_nodes=num_nodes, edge_index=edge_index)

    # solution = AntInspiredHeuristics()._solve(num_nodes, edge_index)
    # solution = HybridHam()._solve(num_nodes, edge_index)
    # solution = least_degree_first_heuristics(num_nodes, edge_index, is_use_unreachable_vertex_heuristics=True)
    # print(solution)

    m = AntInspiredHeuristics()
    evals = eval.EvaluationScores.evaluate_model_on_saved_data(m, 1000, is_show_progress=True)
    acc = eval.EvaluationScores.compute_accuracy_scores(evals)
    print(acc)
