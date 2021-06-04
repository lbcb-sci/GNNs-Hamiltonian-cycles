import itertools
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torch_geometric as torch_g
import torch_scatter

from src.DatasetBuilder import ErdosRenyiInMemoryDataset


class EvaluationScores:
    class ACCURACY_SCORE_TAGS:
        perc_hamilton_found = "perc_hamilton_found"
        perc_long_cycles_found = "perc_long_cycles_found"
        perc_full_walks_found = "perc_full_walks_found"
        perc_long_walks_found = "perc_long_walks_found"

    @staticmethod
    def __format_evaluation_scores_list(score):
        x = torch.cat(score, dim=0)
        return x[torch.nonzero(x).squeeze(-1)]

    @staticmethod
    def verify_only_neighbor_connections(d: torch_g.data.Batch, batch_of_walks):
        walks = [[n.item() for n in batch_of_walks[i]] for i in range(batch_of_walks.shape[0])]
        correct_termination = [not any([w[i] == -1 and w[i+1] != -1 for i in range(len(w) -1)]) for w in walks]
        walks = [[n for n in t if n != -1] for t in walks]

        neighbor_dict = {}
        for (x, y) in torch.t(d.edge_index):
            x, y = x.item(), y.item()
            if x not in neighbor_dict:
                neighbor_dict[x] = [y]
            else:
                neighbor_dict[x] += [y]

        correct_walks = []
        for walk in walks:
            is_correct = True
            for i in range(len(walk) - 1):
                if walk[i] == walk[i+1]:
                    continue
                if walk[i+1] in neighbor_dict[walk[i]]:
                    continue
                is_correct = False
            correct_walks += [is_correct]
        correct_walks = [correct_walks[i] and correct_termination[i] for i in range(len(correct_walks))]
        return torch.tensor(correct_walks, device=d.edge_index.device)

    @staticmethod
    def batch_evaluate(compute_batch_of_walks_fn, batch_graph_generator, max_batches_to_generate):
        cycle_scores = []
        walk_scores = []
        rel_cycle_len = []
        rel_walk_len = []

        with torch.no_grad():
            for d in itertools.islice(batch_graph_generator, max_batches_to_generate):
                walk, selection = compute_batch_of_walks_fn(d)
                walk, selection = walk.detach(), selection.detach()
                start_node_repeats = torch.isclose(walk, walk[..., 0][..., None])
                is_using_proper_edges = EvaluationScores.verify_only_neighbor_connections(d, walk)
                is_cycle = torch.ge(torch.sum(start_node_repeats, dim=-1), 2)
                walk_len = torch.sum(torch.gt(walk, -0.5), dim=-1)
                walk_len = walk_len - torch.ones_like(walk_len)*is_cycle
                cycle_scores += [torch.flatten(is_cycle * walk_len)]
                walk_scores += [torch.flatten(torch.logical_not(is_cycle) * walk_len)]
                rel_cycle_len += [torch.flatten(
                    is_using_proper_edges * is_cycle * walk_len
                    / torch_scatter.scatter_sum(torch.ones_like(d.batch), d.batch, dim=-1, dim_size=d.num_graphs)
                )]
                rel_walk_len += [torch.flatten(
                    is_using_proper_edges * torch.logical_not(is_cycle) * walk_len
                    / torch_scatter.scatter_sum(torch.ones_like(d.batch), d.batch, dim=-1, dim_size=d.num_graphs)
                )]
            result = {name: EvaluationScores.__format_evaluation_scores_list(tensor_list)
                      for name, tensor_list in zip(["cycle_scores", "walk_scores", "rel_cycle_len", "rel_walk_len"],
                                                   [cycle_scores, walk_scores, rel_cycle_len, rel_walk_len])}
        return result

    @staticmethod
    def compute_accuracy_scores(evals, sizes):
        hamilton_perc = []
        approx_hamilton_perc = []
        full_walk_perc = []
        long_walk_perc = []
        perc_ham_graphs = []

        for e, s in zip(evals, sizes):
            total_walks = len(e["rel_cycle_len"]) + len(e["rel_walk_len"])
            for ham, limit in zip([hamilton_perc, approx_hamilton_perc], [0.999, 0.899]):
                ham += [len([t for t in e["rel_cycle_len"] if t > limit])/total_walks]
            for walk, limit in zip([full_walk_perc, long_walk_perc], [0.999, 0.899]):
                walk += [len([t for t in e["rel_walk_len"] if t > limit])/total_walks]
            perc_ham_graphs += [None if e["nr_hamilton_graphs"] is None else e["nr_hamilton_graphs"] / total_walks]
        return hamilton_perc, approx_hamilton_perc, full_walk_perc, long_walk_perc, perc_ham_graphs

    @staticmethod
    def evaluate_on_saved_data(nn_hamilton, nr_batches_per_size, data_folders=None):
        if data_folders is None:
            data_folders = ["../DATA"]
        size_dict = {}
        generator = ErdosRenyiInMemoryDataset(data_folders)
        for graph, hamilton_cycle in generator:
            if graph.num_nodes in size_dict:
                size_dict[graph.num_nodes] += [(graph, hamilton_cycle)]
            else:
                size_dict[graph.num_nodes] = [(graph, hamilton_cycle, hamilton_cycle)]
        sizes = [s for s in sorted(list(size_dict.keys()))]
        evals = []
        batch_size = 10

        for eval_size in sizes:
            it = (torch_g.data.Batch.from_data_list([x[0] for x in size_dict[eval_size][index: index + batch_size]])
                       for index in range(0, len(size_dict[eval_size]), batch_size))
            print(f"Evaluating on {min(len(size_dict[eval_size]), nr_batches_per_size * batch_size)} graphs of"
                  f" size {eval_size} saved in {data_folders}")
            e = EvaluationScores.batch_evaluate(lambda g: nn_hamilton.batch_run_greedy_neighbor(g), it,
                                                nr_batches_per_size)
            e["nr_hamilton_graphs"] = len(
                [data for data in itertools.islice(size_dict[eval_size], nr_batches_per_size * batch_size)
                 if data[1] is not None and len(data[1]) > 0])
            evals += [e]
        return evals, sizes


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
