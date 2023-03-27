from pathlib import Path
import pandas
import numpy
from tqdm import tqdm
from argparse import ArgumentParser
from collections import defaultdict
import torch_geometric as torch_g
from matplotlib import pyplot as plt
import seaborn
from typing import Callable, Any

from hamgnn.data.InMemoryDataset import ErdosRenyiInMemoryDataset
from hamgnn.data.GraphDataset import GraphExample
from hamgnn.visualisation_tools import display_ER_graph_spring, display_result_on_known_hamilton_graphs, display_accuracies, \
    display_runtimes, display_accuracies_with_respect_to_ham_existence_param, display_ablations
from hamgnn.Evaluation import EvaluationScores
from hamgnn.main_model import load_main_model, load_ablation_main_model, load_ablation_no_hidden_features, load_ablation_no_random_features
import hamgnn.constants as constants
from hamgnn.heuristics import HybridHam, LeastDegreeFirstHeuristics, AntInspiredHeuristics
from hamgnn.ExactSolvers import ConcordeHamiltonSolver
from hamgnn.HamiltonSolver import HamiltonSolver
from hamgnn.nn_modules.hamilton_gnn_utils import HamFinderGNN


COMPARISON_WITH_HEURISTICS_CSV_FILENAME = "comparison_with_heuristics.csv"
COMPARISON_WITH_HEURISTICS_FIGURE_STEM = "comparison_with_heuristics"
SUPERCRITICAL_CSV_FILENAME = "non_critical_regime.csv"
SUPERCRITICAL_FIGURE_STEM = "non_critical_regime"
CRITICAL_REGIME_FIGURE_STEM = "critical_regime"
RUNTIMES_FIGURE_STEM = "runtimes"
HAM_PARAMETER_CHANGE_CSV_FILENAME = "ham_parameter_change.csv"
HAM_PARAMETER_CHANGE_FIGURE_STEM = "ham_parameter_change"
BEAM_SEARCH_FIGURE_STEM = "beam_search"
BEAM_SEARCH_RUNTIMES_FIGURE_STEM = "beam_search_runtimes"
ABLATIONS_CSV_FILENAME = "ablations.csv"
ABLATIONS_FIGURE_STEM = "ablations_performance"

HEURISTIC_SOLVERS_MAP = {
    "Concorde": ConcordeHamiltonSolver(),
    "Least degree first": LeastDegreeFirstHeuristics(),
    "HybridHam": HybridHam(),
    "Ant-inspired": AntInspiredHeuristics(),
}
OUR_MODEL_TAG = "Our model"
BEAM_SEARCH_STEM = "Beam width"
BEAM_SEARCH_DEFAULT_WIDTHS = [2, 3, 5]


def get_beam_search_tag(beam_width):
    return f"{BEAM_SEARCH_STEM} {beam_width}"


def _group_graphs_by_size(dataset):
    size_to_graphs_map = defaultdict(list)
    for graph_example in dataset:
        size_to_graphs_map[graph_example.graph.num_nodes].append(graph_example)
    return size_to_graphs_map


def _group_graphs_by_parameter(dataset, fn_get_parameter_from_graph: Callable[[GraphExample], Any]):
    param_to_graphs_map = defaultdict(list)
    for graph_example in dataset:
        param = fn_get_parameter_from_graph(graph_example)
        param_to_graphs_map[param].append(graph_example)
    return param_to_graphs_map


def _accuracy_confidence_interval(nr_graphs, confidence_probability=0.95):
    assert confidence_probability > 0 and confidence_probability < 1
    return numpy.sqrt(-numpy.log((1 - confidence_probability) / 2) / (2 * nr_graphs))


def generate_graph_previews(model, dataset, output_directory: Path, figure_extension="png"):
    assert figure_extension in ["png", "pdf"], 'Only "png" or "pdf" allowed as extensions'
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
            fig_spring_layout.savefig(output_directory / f"ER_{graph_size}_spring_layout.{figure_extension}")

            fig_circular_layout = display_result_on_known_hamilton_graphs(
                graph_example.graph, teacher_HC, teacher_HC, display_node_labels=False, neural_path_color="black")
            fig_circular_layout.savefig(output_directory / f"ER_{graph_size}_circular_layout.{figure_extension}")

            fig_circular_with_ham = display_result_on_known_hamilton_graphs(
                graph_example.graph, nn_solution, teacher_HC, display_node_labels=False, neural_path_color="red", remaining_edges_style="dotted")
            fig_circular_with_ham.savefig(output_directory / f"ER_{graph_size}_circular_layout_with_HC.{figure_extension}")
            break


def _get_default_figure_and_axis():
    return plt.subplots()


def _get_default_colors():
    return seaborn.color_palette()


def _get_default_markers():
    return ["o", ">", "^", "<", "v"]


def _get_default_line_styles():
    return [f"{marker}-" for marker in _get_default_markers()]


def _save_figure(fig, figure_path, format="png"):
    if format not in ["png", "pdf"]:
        print(f'Found "{format}" as required image output format. Only "pdf" or "png" are currently supported. Defaulting to "png"')
        format = "png"
    fig.savefig(figure_path, format=format)


def _load_or_generate_accuracy_data_if_missing(name_to_solver_map, dataset, csv_path: Path):
    size_to_graphs_map = _group_graphs_by_size(dataset)
    _sizes = [size for size in size_to_graphs_map.keys()]
    _nr_graphs = [len(graphs) for graphs in size_to_graphs_map.values()]
    assert [nr == _nr_graphs[0] for nr in _nr_graphs], f"Currently only works if there are equally many graphs for each size. Counted {_sizes}"

    _all_df = []
    if csv_path.exists():
        print(f"Found some measurements in {csv_path}. Will only evaluate missing models")
        df_existing = pandas.read_csv(csv_path)
        _all_df.append(df_existing)
        missing_name_to_solver_map = {name: solver for name, solver in name_to_solver_map.items() if name not in df_existing["name"].unique()}
    else:
        missing_name_to_solver_map = name_to_solver_map.copy()
    if len(missing_name_to_solver_map) > 0:
        eval_graphs_list = []
        for _, graphs in size_to_graphs_map.items():
            eval_graphs_list.extend([graph_example.graph for graph_example in graphs])
        names, solvers = zip(*(missing_name_to_solver_map.items()))
        print(f"Evaluating {len(missing_name_to_solver_map)} solvers: {names}")
        df_new_data = EvaluationScores.accuracy_scores_per_model(solvers, names, eval_graphs_list, is_show_progress=True)
        _all_df.append(df_new_data)
        print("Evaluation complete!")
    df_combined = pandas.concat(_all_df, axis="index").reset_index(drop=True)
    df_combined.to_csv(csv_path, index=False)
    for size in df_combined["size"].unique():
        df_combined.loc[df_combined["size"] == size, constants.PLOTS_CONFIDENCE_DELTA_TAG] = _accuracy_confidence_interval(len(size_to_graphs_map[size]))
    df_combined = df_combined[df_combined["name"].isin(name_to_solver_map.keys())]
    return df_combined


def generate_comparison_plot(model, dataset, output_directory, name_to_solver_map=None, figure_extension="png"):
    if name_to_solver_map is None:
        name_to_solver_map = HEURISTIC_SOLVERS_MAP.copy()
    name_to_solver_map[OUR_MODEL_TAG] = model

    output_directory = Path(output_directory)
    csv_path = output_directory / COMPARISON_WITH_HEURISTICS_CSV_FILENAME

    df_results = _load_or_generate_accuracy_data_if_missing(name_to_solver_map, dataset, csv_path)

    figure_path = output_directory / f"{COMPARISON_WITH_HEURISTICS_FIGURE_STEM}.{figure_extension}"
    fig, ax = _get_default_figure_and_axis()
    nr_solvers = len(df_results["name"].unique())
    display_accuracies(df_results, ax, _get_default_colors(), _get_default_line_styles())
    ax.set_title("HCP performance")
    _save_figure(fig, figure_path=figure_path, format=figure_extension)
    return fig


def generate_supercritical_plot(dataset, output_directory, figure_extension="png"):
    name_to_solver_map = {name: solver for name, solver in HEURISTIC_SOLVERS_MAP.copy().items() if name in ["Concorde", "HybridHam"]}
    output_directory = Path(output_directory)
    csv_path = output_directory / SUPERCRITICAL_CSV_FILENAME

    df_noncritical = _load_or_generate_accuracy_data_if_missing(name_to_solver_map=name_to_solver_map, dataset=dataset, csv_path=csv_path)

    figure_path = output_directory / f"{SUPERCRITICAL_FIGURE_STEM}.{figure_extension}"
    fig, ax = _get_default_figure_and_axis()
    display_accuracies(df_noncritical[df_noncritical["name"].isin(name_to_solver_map)], ax, colors=_get_default_colors(), line_styles=_get_default_line_styles())
    ax.set_title("Supercritical regime")
    _save_figure(fig, figure_path=figure_path, format=figure_extension)
    return fig


def generate_critical_regime_quality_plot(dataset, output_directory, figure_extension):
    csv_path = csv_path=output_directory / COMPARISON_WITH_HEURISTICS_CSV_FILENAME
    df = _load_or_generate_accuracy_data_if_missing({"Concorde": None}, dataset, csv_path=csv_path)
    df = df[df["name"] == "Concorde"]
    fig, ax = _get_default_figure_and_axis()
    display_accuracies(df, ax, _get_default_colors(), _get_default_line_styles())
    ax.set_title("Critical regime")
    figure_path = output_directory / f"{CRITICAL_REGIME_FIGURE_STEM}.{figure_extension}"
    _save_figure(fig, figure_path, figure_extension)
    return fig


def generate_plot_of_runtimes(model, dataset, output_directory, figure_extension="png"):
    csv_path = output_directory / COMPARISON_WITH_HEURISTICS_CSV_FILENAME
    name_to_solver_map = HEURISTIC_SOLVERS_MAP.copy()
    name_to_solver_map["Our model"] = model
    df_test_results = _load_or_generate_accuracy_data_if_missing(name_to_solver_map, dataset, csv_path)

    fig_path = output_directory / f"{RUNTIMES_FIGURE_STEM}.{figure_extension}"
    fig, ax = _get_default_figure_and_axis()
    display_runtimes(df_test_results, ax, colors=_get_default_colors(), markers=_get_default_markers())
    ax.set_title("Runtime (ms)")
    _save_figure(fig, fig_path, format=figure_extension)
    return fig


def generate_plot_of_ham_parametere_changes(model: HamiltonSolver, output_directory, figure_extension="png"):
    csv_path = output_directory / HAM_PARAMETER_CHANGE_CSV_FILENAME

    dataset = ErdosRenyiInMemoryDataset([constants.GRAPH_DATA_DIRECTORY_HAM_PROB_GENERALISATION])
    ham_ex_prob_to_graphs_map = _group_graphs_by_parameter(dataset, lambda graph_example: graph_example.hamilton_existence_probability)

    if csv_path.exists():
        df_results = pandas.read_csv(csv_path)
    else:
        graphs, ham_existence_prob = zip(*[(graph_example.graph, graph_example.hamilton_existence_probability) for graph_example in  dataset])
        solutions = model.timed_solve_graphs(graphs, is_show_progress=True)[0]
        eval = EvaluationScores.evaluate(graphs=graphs, solutions=solutions)
        eval["ham_existence_prob"] = ham_existence_prob
        df_scores = EvaluationScores.compute_scores(eval)
        _df_per_group = []
        for group_name, group in df_scores.groupby("ham_existence_prob"):
            _df_acc = EvaluationScores._compute_accuracy_from_scores(group)
            _df_acc[constants.PLOTS_CONFIDENCE_DELTA_TAG] = len(group)
            _df_acc["name"] = group_name
            _df_acc["hamilton_existence_probability"] = group_name
            _df_per_group.append(_df_acc)
        df_results = pandas.concat(_df_per_group).reset_index(drop=True)
        df_results["name"] = OUR_MODEL_TAG
        df_results.to_csv(csv_path, index=False)

    for ham_ex_prob in df_results["hamilton_existence_probability"].unique():
        df_results.loc[df_results["hamilton_existence_probability"] == ham_ex_prob, constants.PLOTS_CONFIDENCE_DELTA_TAG] = \
            _accuracy_confidence_interval(len(ham_ex_prob_to_graphs_map[ham_ex_prob]))
    figure_path = output_directory / f"{HAM_PARAMETER_CHANGE_FIGURE_STEM}.{figure_extension}"
    fig, ax = _get_default_figure_and_axis()
    display_accuracies_with_respect_to_ham_existence_param(df_results, ax, _get_default_colors(), ["o-" for _ in range(len(df_results["name"].unique()))])
    _save_figure(fig, figure_path=figure_path, format=figure_extension)
    return fig


class BeamSearchWrapper(HamiltonSolver):
    def __init__(self, ham_gnn_model: HamFinderGNN, beam_width) -> None:
        super().__init__()
        self.ham_gnn_model = ham_gnn_model
        self.beam_width = beam_width

    def solve_graphs(self, graphs: list[torch_g.data.Data]) -> list[list[int]]:
        results = []
        for g in graphs:
            results.append(self.ham_gnn_model.run_beam_search(g, beam_width=self.beam_width))
        return results


def generate_beam_search_plot(main_model, dataset, output_directory, figure_extension, beam_widths=None):
    if beam_widths is None:
        beam_widths = BEAM_SEARCH_DEFAULT_WIDTHS
    name_to_solver_map = {OUR_MODEL_TAG: main_model}
    for width in beam_widths:
        beam_model = BeamSearchWrapper(main_model, width)
        name_to_solver_map[get_beam_search_tag(width)] = beam_model

    output_directory = Path(output_directory)
    csv_path = output_directory / COMPARISON_WITH_HEURISTICS_CSV_FILENAME

    df_results = _load_or_generate_accuracy_data_if_missing(name_to_solver_map, dataset, csv_path)

    figure_path = output_directory / f"{BEAM_SEARCH_FIGURE_STEM}.{figure_extension}"
    fig, ax = _get_default_figure_and_axis()
    display_accuracies(df_results, ax, _get_default_colors(), _get_default_line_styles())
    ax.set_title("Performance of different beam widths")
    _save_figure(fig, figure_path=figure_path, format=figure_extension)
    return fig


def generate_plot_of_beam_search_runtimes(main_model, dataset, output_directory, figure_extension="png", beam_widths=None):
    if beam_widths is None:
        beam_widths = BEAM_SEARCH_DEFAULT_WIDTHS
    name_to_solver_map = {OUR_MODEL_TAG: main_model}
    for width in beam_widths:
        beam_model = BeamSearchWrapper(main_model, width)
        name_to_solver_map[get_beam_search_tag(width)] = beam_model

    output_directory = Path(output_directory)
    csv_path = output_directory / COMPARISON_WITH_HEURISTICS_CSV_FILENAME

    df_results = _load_or_generate_accuracy_data_if_missing(name_to_solver_map, dataset, csv_path)

    figure_path = output_directory / f"{BEAM_SEARCH_RUNTIMES_FIGURE_STEM}.{figure_extension}"
    fig, ax = _get_default_figure_and_axis()
    display_runtimes(df_results, ax, _get_default_colors(), _get_default_markers(), ceil_on_time_axis=None)
    ax.set_title("Runtime of different beam widths")
    _save_figure(fig, figure_path=figure_path, format=figure_extension)
    return fig


def generate_ablation_plot(dataset, output_directory, figure_extension, nr_retrains=5):
    main_models = {f"Main model_{i}": load_ablation_main_model(i) for i in range(nr_retrains)}
    no_hidden = {f"No persistent features_{i}": load_ablation_no_hidden_features(i) for i in range(nr_retrains)}
    no_random = {f"No random features_{i}": load_ablation_no_random_features(i) for i in range(nr_retrains)}
    name_to_solver_map = {}
    name_to_solver_map.update(main_models)
    name_to_solver_map.update(no_hidden)
    name_to_solver_map.update(no_random)

    output_directory = Path(output_directory)
    csv_path = output_directory / ABLATIONS_CSV_FILENAME

    df_results = _load_or_generate_accuracy_data_if_missing(name_to_solver_map, dataset, csv_path)

    figure_path = output_directory / f"{ABLATIONS_FIGURE_STEM}.{figure_extension}"
    fig, ax = _get_default_figure_and_axis()

    df_results["class"] = df_results["name"].apply(lambda name: "_".join(name.split("_")[:-1]))
    display_ablations(df_results, ax, _get_default_colors(), _get_default_line_styles())
    _save_figure(fig, figure_path=figure_path, format=figure_extension)
    return fig


if __name__ == '__main__':
    parser = ArgumentParser("Generates figures presented in the papaer")
    parser.add_argument("output_dir", type=str, help="Directory where images are stored")
    parser.add_argument("--ext", type=str, help="Format of generated images, either png or pdf", default="png")
    args = parser.parse_args()
    output_directory = Path(args.output_dir)
    figure_extension = args.ext

    if not output_directory.exists():
        output_directory.mkdir(parents=True)

    dataset = ErdosRenyiInMemoryDataset([constants.GRAPH_DATA_DIRECTORY_SIZE_GENERALISATION])
    main_model = load_main_model()

    generate_graph_previews(main_model, dataset, output_directory=output_directory, figure_extension=figure_extension)
    with plt.style.context("seaborn-paper"):
        generate_comparison_plot(main_model, dataset,output_directory=output_directory, figure_extension=figure_extension)
        generate_plot_of_runtimes(main_model, dataset, output_directory=output_directory, figure_extension=figure_extension)
        supercritical_dataset = ErdosRenyiInMemoryDataset([constants.GRAPH_DATA_DIRECTORY_SUPERCRITICAL_CASE])
        generate_supercritical_plot(supercritical_dataset, output_directory=output_directory, figure_extension=figure_extension)
        generate_critical_regime_quality_plot(dataset, output_directory=output_directory, figure_extension=figure_extension)
        generate_plot_of_ham_parametere_changes(main_model, output_directory=output_directory, figure_extension=figure_extension)
        generate_beam_search_plot(main_model, dataset, output_directory, figure_extension=figure_extension)
        generate_plot_of_beam_search_runtimes(main_model, dataset, output_directory, figure_extension)
        generate_ablation_plot(dataset, output_directory, figure_extension)
