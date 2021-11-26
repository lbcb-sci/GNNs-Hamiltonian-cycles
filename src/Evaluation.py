import matplotlib.pyplot as plt
import pandas
import numpy
from typing import List

import torch
import torch.utils.data
import torch_geometric as torch_g

from src.DatasetBuilder import ErdosRenyiInMemoryDataset
from src.Models import HamFinderGNN
from src.constants import *


class EvaluationScores:
    class ACCURACY_SCORE_TAGS:
        perc_hamilton_found = "perc_hamilton_found"
        perc_long_cycles_found = "perc_long_cycles_found"
        perc_full_walks_found = "perc_full_walks_found"
        perc_long_walks_found = "perc_long_walks_found"

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
    def evaluate(solve_graphs, graph_list: List[torch_g.data.Data]):
        walks = solve_graphs(g for g in graph_list)
        is_valid = [not EvaluationScores.is_walk_valid(graph, walk) for graph, walk in zip(graph_list, walks)]
        is_cycle = [walk[0] == walk[-1] for walk in walks]
        nr_unique_nodes = [len(set(walk)) for walk in walks]
        sizes = [graph.num_nodes for graph in graph_list]

        return {"is_cycle": is_cycle, "is_valid": is_valid, "length": nr_unique_nodes, "size": sizes}

    @staticmethod
    def compute_accuracy_scores(evals):
        df = pandas.DataFrame.from_dict(evals)
        df["is_ham_cycle"] = (df["length"] == df["size"]) & (df["is_cycle"])
        df["is_ham_path"] = (df["length"] == df["size"]) & (~df["is_cycle"])
        df["is_approx_ham_cycle"] \
            = (df["length"] > EvaluationScores.APPROXIMATE_HAMILTON_LOWER_BOUND * df["size"]) & (df["is_cycle"])
        df["is_approx_ham_path"] \
            = (df["length"] > EvaluationScores.APPROXIMATE_HAMILTON_LOWER_BOUND * df["size"]) & (~df["is_cycle"])

        measurement_columns = ["is_ham_cycle", "is_ham_path", "is_approx_ham_cycle", "is_approx_ham_path"]
        grouped = df[["size"] + measurement_columns].groupby("size").aggregate({name: "mean" for name in measurement_columns}).reset_index()
        grouped.columns = [col.replace("is_", "perc_") for col in grouped]
        return grouped

    @staticmethod
    def _filtered_generator(num_per_size, gen):
        size_counter = {}
        for (graph, ham_cycle) in gen:
            if graph.num_nodes in size_counter:
                size_counter[graph.num_nodes] += 1
            else:
                size_counter[graph.num_nodes] = 1
            if size_counter[graph.num_nodes] > num_per_size:
                continue
            else:
                yield (graph, ham_cycle)

    @staticmethod
    def evaluate_on_saved_data(compute_walks_from_graph_list_fn, nr_graphs_per_size=10, data_folders=None):
        if data_folders is None:
            data_folders = EVALUATION_DATA_FOLDERS

        def _get_generator():
            gen = ErdosRenyiInMemoryDataset(data_folders)
            if nr_graphs_per_size is not None:
                gen = EvaluationScores._filtered_generator(nr_graphs_per_size, gen)
            return gen

        is_hamiltonian = numpy.array([x[1] is not None and len(x[1]) > 0 for x in _get_generator()])

        graph_list = [graph for graph, ham_cycle in _get_generator()]
        evals = EvaluationScores.evaluate(compute_walks_from_graph_list_fn, graph_list)
        evals["is_graph_hamiltonian"] = is_hamiltonian
        return evals

    @staticmethod
    def evaluate_model_on_saved_data(nn_hamilton: HamFinderGNN, nr_graphs_per_size=10, data_folders=None):
        def _compute_walks_from_graph_list_fn(graph_list):
            return nn_hamilton.solve_graphs(graph_list)

        return EvaluationScores.evaluate_on_saved_data(_compute_walks_from_graph_list_fn, nr_graphs_per_size, data_folders)


class EvaluationPlots:
    DEFAULT_FIGSIZE = (16, 9)

    @staticmethod
    def single_eval_hist(eval_results, window_title):
        fig = plt.figure(figsize=EvaluationPlots.DEFAULT_FIGSIZE)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, sharey=ax1)
        fig.canvas.set_window_title(window_title)
        ax1.set_title("Relative cycle length")
        ax1.hist(eval_results["rel_cycle_len"], 20)
        ax2.set_title("Relative walk length")
        ax2.hist(eval_results["rel_walk_len"], 20)
        return fig

    @staticmethod
    def combined_histograms(evals, sizes):
        fig = plt.figure(figsize=EvaluationPlots.DEFAULT_FIGSIZE)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.hist([e["rel_cycle_len"] for e in evals], 10, label=["ER-{}".format(s) for s in sizes])
        ax1.set_title("Cycles")
        ax1.set_ylabel("nr in Erdos-Renyi grap")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.hist([e["rel_walk_len"] for e in evals], 10)
        ax2.set_title("Walks (start != finish)")
        for a in [ax1, ax2]:
            a.set_xlabel("relative len")
            fig.legend(loc="upper left")
        return fig

    @staticmethod
    def list_of_histograms(evals, sizes):
        fig, axes = plt.subplots(nrows=len(evals), ncols=2, sharex=True, sharey=True)
        ax_count = 1
        for ax_index, size, eval_result in zip(range(len(sizes)), sizes, evals):
            ax1 = axes[ax_index, 0]
            ax2 = axes[ax_index, 1]
            if ax_index == 0:
                ax1.set_title("Relative cycle length")
                ax2.set_title("Relative walk length")
            ax1.set_ylabel("nr in {}-ER".format(size))
            ax_count += 2
            ax1.hist(eval_result["rel_cycle_len"], 20)
            ax2.hist(eval_result["rel_walk_len"], 20)
        return fig

    @staticmethod
    def accuracy_curves(evals, sizes, best_expected_benchmark=None):
        hamilton_perc, approx_hamilton_perc, full_walk_perc, long_walk_perc, perc_ham_graphs \
            = EvaluationScores.compute_accuracy_scores(evals, sizes)
        fig = plt.figure(figsize=EvaluationPlots.DEFAULT_FIGSIZE)
        ax = fig.add_subplot(1, 1, 1)
        for line, color, marker, label in zip([hamilton_perc, approx_hamilton_perc, full_walk_perc, long_walk_perc],
                                              ["red", "orange", "blue", "cyan"], [".", "v", "x", "D"],
                                              ["Hamilton cycles found", "Cycles with > 90% nodes found",
                                               "Complete walks (start != finish) found", "Walks with > 90% nodes found"]):
            ax.plot(sizes, [x*100 for x in line], color=color, label=label, marker=marker, linestyle='dotted', markersize=12)
        if all([x is not None for x in perc_ham_graphs]):
            ax.plot(sizes, [x*100 for x in perc_ham_graphs], color="gray", label="Hamiltonian graphs", linestyle="dashed")
        elif best_expected_benchmark is not None:
            ax.plot(sizes, [best_expected_benchmark*100 for _ in sizes], linestyle="dashed", label="Expected Hamilton cycles", color="gray")
        ax.set_xlabel("Graph size")
        ax.set_ylabel("% of solutions")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right")
        return fig

    @staticmethod
    def model_performance(evals):
        pass
