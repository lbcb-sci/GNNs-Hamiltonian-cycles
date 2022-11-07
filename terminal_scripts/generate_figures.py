from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from collections import defaultdict
import torch_geometric as torch_g
from matplotlib import pyplot as plt

from hamgnn.data.InMemoryDataset import ErdosRenyiInMemoryDataset
from hamgnn.visualisation_tools import display_ER_graph_spring, display_result_on_known_hamilton_graphs
from hamgnn.Evaluation import EvaluationScores
from hamgnn.main_model import load_main_model
import hamgnn.constants as constants


def generate_graph_previews(model, dataset, output_directory: Path):
    graph_size_to_data_dict = defaultdict(list)
    for graph_example in dataset:
        graph_size_to_data_dict[graph_example.graph.num_nodes].append(graph_example)

    sizes = sorted(list(graph_size_to_data_dict.keys()))

    for graph_size in sizes[:3]:
        for try_nr, graph_example in tqdm(enumerate(graph_size_to_data_dict[graph_size]), desc=f"Solving graphs until Hamiltonian cycle is found", leave=True):
            if graph_example.hamiltonian_cycle is None:
                continue
            teacher_HC = [x.item() for x in graph_example.hamiltonian_cycle]
            nn_solution = model.solve(graph_example.graph)
            if len(nn_solution) < graph_size + 1 or nn_solution[0] != nn_solution[-1]:
                continue
            valid_path = EvaluationScores.is_walk_valid(graph_example.graph, nn_solution)
            if not valid_path:
                raise Exception("Something is wrong with solution algorithm, it produces invalid path")
            fig_spring_layout = display_ER_graph_spring(graph_example.graph)
            fig_spring_layout.savefig(output_directory / f"ER_{graph_size}_spring_layout.png")

            fig_circular_layout = display_result_on_known_hamilton_graphs(
                graph_example.graph, teacher_HC, teacher_HC, display_node_labels=False, neural_path_color="black")
            fig_circular_layout.savefig(output_directory / f"ER_{graph_size}_circular_layout.png")

            fig_circular_with_ham = display_result_on_known_hamilton_graphs(
                graph_example.graph, nn_solution, teacher_HC, display_node_labels=False, neural_path_color="red", remaining_edges_style="dotted")
            fig_circular_with_ham.savefig(output_directory / f"ER_{graph_size}_circular_layout_with_HC.png")
            break


if __name__ == '__main__':
    parser = ArgumentParser("Generates figures presented in the papaer")
    parser.add_argument("output_dir", type=str, help="Directory where images are stored")
    args = parser.parse_args()
    output_directory = Path(args.output_dir)
    if not output_directory.exists():
        output_directory.mkdir(parents=True)

    dataset = ErdosRenyiInMemoryDataset([constants.GRAPH_DATA_DIRECTORY_SIZE_GENERALISATION])
    main_model = load_main_model()

    generate_graph_previews(main_model, dataset, output_directory=output_directory)
