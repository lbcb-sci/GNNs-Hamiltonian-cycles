from hamgnn.data.GraphGenerators import ErdosRenyiGenerator
import itertools
import torch_geometric
import networkx
from matplotlib import pyplot as plt
from copy import deepcopy


from hamgnn.heuristics import *


if __name__ == '__main__':
    from pathlib import Path
    path = Path(__file__).parent.parent.parent / "HCP_benchmarks/graph2.hcp"
    num_nodes, edge_index = load_graph_from_hcp_file(path)
    graph = torch_geometric.data.Data(num_nodes=num_nodes, edge_index=edge_index)
    # solution = HybridHam()._solve(num_nodes, edge_index)
    # solution = least_degree_first_heuristics(num_nodes, edge_index, is_use_unreachable_vertex_heuristics=True)
    # print(solution)

    m = HybridHam()
    import hamgnn.Evaluation as eval
    evals = eval.EvaluationScores.evaluate_model_on_saved_data(m, 10000)
    acc = eval.EvaluationScores.compute_accuracy_scores(evals)


    num_nodes = 100
    generator = ErdosRenyiGenerator(num_nodes, 0.4)

    for d in itertools.islice(generator, 100):
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
