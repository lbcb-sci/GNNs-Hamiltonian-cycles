import torch_geometric as torch_g
from matplotlib import pyplot as plt
from src.DatasetBuilder import ErdosRenyiInMemoryDataset
from src.VisualisationTools import display_ER_graph, display_result_on_known_hamilton_graphs

if __name__ == '__main__':
    from train import train_HamS, EVALUATION_DATA_FOLDERS
    from src.Evaluation import EvaluationScores
    hamS = train_HamS(True, 0)
    dataset = ErdosRenyiInMemoryDataset(EVALUATION_DATA_FOLDERS)

    graph_size_to_data_dict = {}
    for d, c in dataset:
        if d.num_nodes in graph_size_to_data_dict:
            graph_size_to_data_dict[d.num_nodes] += [(d,c)]
        else:
            graph_size_to_data_dict[d.num_nodes] = [(d,c)]

    for graph_size in [25, 50]:
        if graph_size in graph_size_to_data_dict:
            d, cycle = graph_size_to_data_dict[graph_size][0]
            fig_spring_layout = display_ER_graph(d)
            fig_spring_layout.show()
            if cycle is None:
                print("The graph is not Hamiltonian")
                continue

            fig_circular_layout = display_result_on_known_hamilton_graphs(d, cycle, cycle, display_node_labels=False,
                                                          neural_path_color="black")
            fig_circular_layout.show()
            nn_path, distributions = hamS.batch_run_greedy_neighbor(torch_g.data.Batch.from_data_list([d]))
            valid_path = EvaluationScores.verify_only_neighbor_connections(d, nn_path)
            nn_path = [x.item() for x in nn_path.flatten()]
            if len(nn_path) != graph_size + 1 or not valid_path or nn_path[0] != nn_path[-1]:
                print("Neural network failed to find a Hamilton cycle!")
                continue
            fig_predicted_cylce = display_result_on_known_hamilton_graphs(d, nn_path, cycle, display_node_labels=False,
                                                                          neural_path_color="red", remaining_edges_style="dotted")
            fig_predicted_cylce.show()
            plt.show()

        else:
            print(f"Failed to find any graphs of size {graph_size} in {EVALUATION_DATA_FOLDERS}!")
            continue
