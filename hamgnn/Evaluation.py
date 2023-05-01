import pandas
import numpy
from typing import List
import seaborn
from tqdm import tqdm

import torch
import torch.utils.data
import torch_geometric as torch_g

import hamgnn.data.InMemoryDataset as InMemoryDataset
from hamgnn.HamiltonSolver import HamiltonSolver
from hamgnn.constants import *


class EvaluationScores:
    class ACCURACY_SCORE_TAGS:
        perc_hamilton_found = "perc_hamilton_found"
        perc_long_cycles_found = "perc_long_cycles_found"
        perc_full_walks_found = "perc_full_walks_found"
        perc_long_walks_found = "perc_long_walks_found"
        avg_execution_time = "avg_execution_time"

    APPROXIMATE_HAMILTON_LOWER_BOUND = 0.9

    @staticmethod
    def is_walk_valid(graph: torch_g.data.Data, walk):
        neighbor_dict = {x: [] for x in range(graph.num_nodes)}
        for (x, y) in torch.t(graph.edge_index):
            x, y = x.item(), y.item()
            neighbor_dict[x].append(y)

        visited_nodes = set()
        for i in range(len(walk) - 1):
            visited_nodes.add(walk[i])
            if walk[i + 1] in visited_nodes:
                return True if (i == len(walk) - 2) else False
            if walk[i + 1] not in neighbor_dict[walk[i]]:
                return False
        return True

    @staticmethod
    def evaluate(graphs: list[torch_g.data.Data], solutions: list[int]):
        is_valid = numpy.array([EvaluationScores.is_walk_valid(graph, walk) for graph, walk in zip(graphs, solutions)])
        is_cycle = numpy.array([len(walk) > 0 and walk[0] == walk[-1] for walk in solutions]) & is_valid
        nr_unique_nodes = [len(set(walk)) for walk in solutions]
        sizes = [graph.num_nodes for graph in graphs]

        return {"is_cycle": is_cycle, "is_valid": is_valid, "length": nr_unique_nodes, "size": sizes}

    @staticmethod
    def solve_time_and_evaluate(timed_solve_graphs, graphs: List[torch_g.data.Data], is_show_progress=False):
        solutions, times = timed_solve_graphs(list(g for g in graphs), is_show_progress)
        evals = EvaluationScores.evaluate(graphs, solutions)
        evals["time"] = times
        return evals

    @staticmethod
    def compute_scores(evals):
        df = pandas.DataFrame.from_dict(evals)
        df["is_ham_cycle"] = (df["length"] == df["size"]) & (df["is_cycle"]) & df["is_valid"]
        df["is_ham_path"] = (df["length"] == df["size"]) & (~df["is_cycle"]) & df["is_valid"]
        df["is_approx_ham_cycle"] \
            = (df["length"] > EvaluationScores.APPROXIMATE_HAMILTON_LOWER_BOUND * df["size"]) & (df["is_cycle"])
        df["is_approx_ham_path"] \
            = (df["length"] > EvaluationScores.APPROXIMATE_HAMILTON_LOWER_BOUND * df["size"]) & (~df["is_cycle"])
        return df

    @staticmethod
    def _compute_accuracy_from_scores(df_scores):
        measurement_columns = ["is_ham_cycle", "is_ham_path", "is_approx_ham_cycle", "is_approx_ham_path", "time"]
        present_measurement_columns = [c for c in measurement_columns if c in df_scores]
        df_scores = df_scores[["size"] + present_measurement_columns].groupby("size").aggregate({name: "mean" for name in present_measurement_columns}).reset_index()
        _columns_rename_dict = {
            "is_ham_cycle": EvaluationScores.ACCURACY_SCORE_TAGS.perc_hamilton_found,
            "is_ham_path": EvaluationScores.ACCURACY_SCORE_TAGS.perc_full_walks_found,
            "is_approx_ham_cycle": EvaluationScores.ACCURACY_SCORE_TAGS.perc_long_cycles_found,
            "is_approx_ham_path": EvaluationScores.ACCURACY_SCORE_TAGS.perc_long_walks_found,
            "time": EvaluationScores.ACCURACY_SCORE_TAGS.avg_execution_time
        }
        return df_scores.rename(columns=_columns_rename_dict)

    @staticmethod
    def compute_accuracy_scores(evals):
        df = EvaluationScores.compute_scores(evals)
        return EvaluationScores._compute_accuracy_from_scores(df)

    @staticmethod
    def _filtered_generator(num_per_size, gen):
        size_counter = {}
        for graph_example in gen:
            graph = graph_example.graph
            ham_cycle = graph_example.teacher_path
            if graph.num_nodes in size_counter:
                size_counter[graph.num_nodes] += 1
            else:
                size_counter[graph.num_nodes] = 1
            if num_per_size is not None and size_counter[graph.num_nodes] > num_per_size:
                continue
            else:
                yield (graph, ham_cycle)

    @staticmethod
    def accuracy_scores_per_model(models: List[HamiltonSolver], names: List[str], graphs: List[torch_g.data.Data], is_show_progress=False):
        all_dfs = []
        progress_bar = list(zip(models, names))
        if is_show_progress:
            progress_bar = tqdm(progress_bar)
        for model, name in progress_bar:
            evals = EvaluationScores.solve_time_and_evaluate(lambda graphs, is_show_progress: model.timed_solve_graphs(graphs, is_show_progress),
                                                             graphs, is_show_progress=is_show_progress)
            _df = EvaluationScores.compute_accuracy_scores(evals)
            _df["name"] = name
            all_dfs.append(_df)
        df_combined = pandas.concat(all_dfs, axis="index").reset_index(drop=True)
        return df_combined


    @staticmethod
    def evaluate_on_saved_data(compute_walks_from_graph_list_fn, nr_graphs_per_size=10, data_folders=None, is_show_progress=False):
        if data_folders is None:
            data_folders = [GRAPH_DATA_DIRECTORY_SIZE_GENERALISATION]

        def _get_generator():
            graph_example_gen = InMemoryDataset.ErdosRenyiInMemoryDataset(data_folders)
            return EvaluationScores._filtered_generator(nr_graphs_per_size, graph_example_gen)

        is_hamiltonian = numpy.array([x[1] is not None and len(x[1]) > 0 for x in _get_generator()])

        graph_list = [graph for graph, ham_cycle in _get_generator()]
        evals = EvaluationScores.solve_time_and_evaluate(compute_walks_from_graph_list_fn, graph_list, is_show_progress)
        evals["is_graph_hamiltonian"] = is_hamiltonian
        return evals

    @staticmethod
    def evaluate_model_on_saved_data(nn_hamilton: HamiltonSolver, nr_graphs_per_size=10, data_folders=None, is_show_progress=False):
        def _compute_walks_from_graph_list_fn(graph_list, _is_show_progres):
            return nn_hamilton.timed_solve_graphs(graph_list, _is_show_progres)

        return EvaluationScores.evaluate_on_saved_data(_compute_walks_from_graph_list_fn, nr_graphs_per_size, data_folders, is_show_progress)

    @staticmethod
    def accuracy_scores_on_saved_data(solvers: List[HamiltonSolver], solver_names: List[str], nr_graphs_per_size=10, data_folders=None, best_possible_score=None, is_show_progress=False):
        evaluations_list = []
        for solver, name in zip(solvers, solver_names):
            evals = EvaluationScores.evaluate_model_on_saved_data(solver, nr_graphs_per_size=nr_graphs_per_size, data_folders=data_folders, is_show_progress=is_show_progress)
            _df_solver_score = pandas.DataFrame(EvaluationScores.compute_accuracy_scores(evals))
            _df_solver_score["name"] = name
            evaluations_list.append(_df_solver_score)
        df_scores = pandas.concat(evaluations_list, axis=0)
        df_scores = df_scores.rename(columns={"size": "graph size"})
        return df_scores


class EvaluationPlots:
    DEFAULT_FIGSIZE = (16, 9)
    @staticmethod
    def accuracy_curves_from_scores(df_solver_scores, columns_containing_scores_to_titles = None, score_axis_label="score", x_axis_column_name="graph size"):
        if columns_containing_scores_to_titles is None:
            columns_containing_scores_to_titles = {
                EvaluationScores.ACCURACY_SCORE_TAGS.perc_hamilton_found: "Hamiltonian cycle",
                EvaluationScores.ACCURACY_SCORE_TAGS.perc_full_walks_found: "Hamiltonian path",
                EvaluationScores.ACCURACY_SCORE_TAGS.perc_long_cycles_found: "long cycle (>90% nodes)",
                EvaluationScores.ACCURACY_SCORE_TAGS.perc_long_walks_found: "long paths (>90% nodes)",
            }
        _extracted_dfs = []
        for score_column, score_name in columns_containing_scores_to_titles.items():
            _score_df = df_solver_scores[[c for c in df_solver_scores if c not in columns_containing_scores_to_titles]].copy()
            _score_df["score_type"] = score_name
            _score_df[score_axis_label] = df_solver_scores[score_column]
            _extracted_dfs.append(_score_df)
        df_plotting = pandas.concat(_extracted_dfs)
        facet_grid = seaborn.FacetGrid(data=df_plotting, col="score_type", hue="name", height=6, margin_titles=True)
        facet_grid.map(seaborn.lineplot, x_axis_column_name, score_axis_label)
        facet_grid.set_titles(col_template="{col_name}")
        facet_grid.add_legend()
        return facet_grid._figure

    @staticmethod
    def model_performance(evals):
        pass
