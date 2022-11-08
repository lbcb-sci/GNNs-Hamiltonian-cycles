from pathlib import Path
import pandas
import numpy
from tqdm import tqdm
from argparse import ArgumentParser
from collections import defaultdict
import torch_geometric as torch_g
from matplotlib import pyplot as plt

from hamgnn.data.InMemoryDataset import ErdosRenyiInMemoryDataset
from hamgnn.visualisation_tools import display_ER_graph_spring, display_result_on_known_hamilton_graphs, display_accuracies
from hamgnn.Evaluation import EvaluationScores
from hamgnn.main_model import load_main_model
import hamgnn.constants as constants
from hamgnn.heuristics import HybridHam, LeastDegreeFirstHeuristics, AntInspiredHeuristics
from hamgnn.ExactSolvers import ConcordeHamiltonSolver


COMPARISON_WITH_HEURISTICS_CSV_FILENAME = "comparison_with_heuristics.csv"
COMPARISON_WITH_HEURISTICS_FIGURE_STEM = "comparison_with_heuristics"


def _group_graphs_by_size(dataset):
    size_to_graphs_map = defaultdict(list)
    for graph_example in dataset:
        size_to_graphs_map[graph_example.graph.num_nodes].append(graph_example)
    return size_to_graphs_map


def _accuracy_confidence_interval(nr_graphs, confidence_probability=0.95):
    assert confidence_probability > 0 and confidence_probability < 1
    return numpy.sqrt(-numpy.log((1 - confidence_probability) / 2) / (2 * nr_graphs))


def generate_graph_previews(model, dataset, output_directory: Path):
    size_to_graphs_map = _group_graphs_by_size(dataset)
    sizes = sorted(list(size_to_graphs_map.keys()))

    for graph_size in sizes[:3]:
        for try_nr, graph_example in tqdm(enumerate(size_to_graphs_map[graph_size]), desc=f"Solving graphs until Hamiltonian cycle is found", leave=True):
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


def generate_comparison_plot(model, dataset, output_directory, name_to_solver_map=None, figure_extension="png"):
    assert figure_extension in ["png", "pdf"], "Only .pdf and .png extensions currently supported"
    if name_to_solver_map is None:
        name_to_solver_map = {
            "concorde": ConcordeHamiltonSolver(),
            "least_degree_frist": LeastDegreeFirstHeuristics(),
            "HybridHam": HybridHam(),
            "ant": AntInspiredHeuristics()
        }
    name_to_solver_map["main_model"] = model

    size_to_graph_map = _group_graphs_by_size(dataset)
    _sizes = [size for size in size_to_graph_map.keys()]
    _nr_graphs = [len(graphs) for graphs in size_to_graph_map.values()]
    assert [nr == _nr_graphs[0] for nr in _nr_graphs], f"Currently only works if there are equally many graphs for each size. Counted {_sizes}"
    nr_graphs_per_size = _nr_graphs[0]

    eval_graphs_list = []
    for _, graphs in size_to_graph_map.items():
        eval_graphs_list.extend([graph_example.graph for graph_example in graphs])

    output_directory = Path(output_directory)
    csv_path = output_directory / COMPARISON_WITH_HEURISTICS_CSV_FILENAME

    _all_df = []
    if csv_path.exists():
        print(f"Found some measurements in {csv_path}. Will only evaluate missing models")
        df_existing = pandas.read_csv(csv_path)
        _all_df.append(df_existing)
        name_to_solver_map = {name: solver for name, solver in name_to_solver_map.items() if name not in df_existing["name"].unique()}
    if len(name_to_solver_map) > 0:
        names, solvers = zip(*(name_to_solver_map.items()))
        print(f"Evaluating {len(name_to_solver_map)} solvers: {names}")
        df_new_data = EvaluationScores.accuracy_scores_per_model(solvers, names, eval_graphs_list, is_show_progress=True)
        _all_df.append(df_new_data)
    print("Evaluation complete!")
    df_combined = pandas.concat(_all_df, axis="index").reset_index(drop=True)
    df_combined.to_csv(csv_path, index=False)
    confidence_delta = _accuracy_confidence_interval(nr_graphs_per_size)
    df_combined["confidence_delta"] = confidence_delta

    figure_path = output_directory / f"{COMPARISON_WITH_HEURISTICS_FIGURE_STEM}.{figure_extension}"
    fig, ax = plt.subplots()
    nr_solvers = len(df_combined["name"].unique())
    display_accuracies(df_combined, ax, ["red", "green", "blue", "yellow", "orange"], ["o--" for _ in range(nr_solvers)])
    fig.savefig(figure_path, format=figure_extension)


if __name__ == '__main__':
    parser = ArgumentParser("Generates figures presented in the papaer")
    parser.add_argument("output_dir", type=str, help="Directory where images are stored")
    args = parser.parse_args()
    output_directory = Path(args.output_dir)
    if not output_directory.exists():
        output_directory.mkdir(parents=True)

    dataset = ErdosRenyiInMemoryDataset([constants.GRAPH_DATA_DIRECTORY_SIZE_GENERALISATION])
    main_model = load_main_model()

    # generate_graph_previews(main_model, dataset, output_directory=output_directory)
    generate_comparison_plot(main_model, dataset,output_directory=output_directory)
