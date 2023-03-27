import pandas
import seaborn
import networkx as nx
import torch_geometric as torch_g
from matplotlib import pyplot as plt
import math

import hamgnn.constants as constants


DEFAULT_FIG_SIZE = (7, 7)

def display_result_on_known_hamilton_graphs(d: torch_g.data.Data, nn_path, hamilton_cycle, display_node_labels=True,
                                            neural_path_color="red", remaining_edges_style="solid"):
    assert len(nn_path) > 0
    g = torch_g.utils.to_networkx(d)
    g = nx.Graph(nx.to_undirected(g))

    pos = {n: p for (_, p), n in zip(nx.circular_layout(g).items(), hamilton_cycle)}
    nn_edges = [(nn_path[i], nn_path[i + 1]) for i in range(len(nn_path) - 1)]

    fig = plt.figure(figsize=DEFAULT_FIG_SIZE)
    ax = fig.add_subplot(1, 1, 1)

    nx.draw(g, pos, ax=ax, with_labels=display_node_labels, style=remaining_edges_style)
    g.remove_edges_from(list(g.edges))
    g.add_edges_from(nn_edges)
    nx.draw_networkx_edges(g, pos, edge_color=neural_path_color, ax=ax)

    return fig


def display_ER_graph_spring(d: torch_g.data):
    g = torch_g.utils.to_networkx(d)
    g = nx.to_undirected(g)
    fig = plt.figure(figsize=DEFAULT_FIG_SIZE)
    ax = fig.add_subplot(1, 1, 1)
    nx.draw(g, nx.spring_layout(g), ax=ax)
    return fig


def display_accuracies(df: pandas.DataFrame, ax, colors=None, line_styles=None, fill_alpha=0.2):
    _unique_sizes = sorted([s for s in df["size"].unique()])
    ax.set_xlabel("Graph size (number of nodes)")
    _x_min = 0
    _x_max = max(_unique_sizes) * 1.1
    ax.set_xlim(_x_min, _x_max)
    ax.set_xticks(_unique_sizes)
    ax.set_xticklabels(_unique_sizes)
    ax.set_ylabel("Fraction of graphs solved (HC found)")
    ax.set_ylim(-0.05, 1.1)
    _yticks = [0.1 * x for x in range(0, 11, 1)]
    ax.set_yticks(_yticks)
    ax.set_yticklabels([f"{x:.1f}" for x in _yticks])

    for idx, (group_name, group) in enumerate(df.groupby("name")):
        line_style = line_styles[idx]
        color = colors[idx]
        ax.plot(group["size"], group["perc_hamilton_found"], line_style, color=color, label=group_name)
        if constants.PLOTS_CONFIDENCE_DELTA_TAG in group:
            ax.fill_between(
                group["size"],
                group["perc_hamilton_found"] + group[constants.PLOTS_CONFIDENCE_DELTA_TAG],
                group["perc_hamilton_found"] - group[constants.PLOTS_CONFIDENCE_DELTA_TAG],
                color=color,
                alpha=fill_alpha)
    ax.legend()


def display_accuracies_with_respect_to_ham_existence_param(df: pandas.DataFrame, ax, colors=None, line_styles=None, fill_alpha=0.2):
    _unique_ham_probs = sorted([s for s in df["hamilton_existence_probability"].unique()])
    ax.set_xlabel("Limit probability for existence of HC")
    _x_min = min(_unique_ham_probs) - 0.1
    _x_max = max(_unique_ham_probs)  + 0.1
    ax.set_xlim(_x_min, _x_max)
    ax.set_xticks(_unique_ham_probs)
    ax.set_xticklabels([f"{p:.2f}" for p in _unique_ham_probs])
    ax.set_ylabel("Fraction of solvable HCPs solved")
    ax.set_ylim(-0.05, 1.1)
    _yticks = [0.1 * x for x in range(0, 11, 1)]
    ax.set_yticks(_yticks)
    ax.set_yticklabels([f"{x:.1f}" for x in _yticks])

    for idx, (group_name, group) in enumerate(df.groupby("name")):
        line_style = line_styles[idx]
        color = colors[idx]
        ax.plot(group["hamilton_existence_probability"], group["perc_hamilton_found"], line_style, color=color, label=group_name)
        if constants.PLOTS_CONFIDENCE_DELTA_TAG in group:
            ax.fill_between(
                group["hamilton_existence_probability"],
                group["perc_hamilton_found"] + group[constants.PLOTS_CONFIDENCE_DELTA_TAG],
                group["perc_hamilton_found"] - group[constants.PLOTS_CONFIDENCE_DELTA_TAG],
                color=color,
                alpha=fill_alpha)
    ax.legend()


def display_runtimes(df: pandas.DataFrame, ax, colors, markers, ceil_on_time_axis=0.4):
    seaborn.set_palette(colors)
    seaborn.lineplot(data=df, x="size", y="avg_execution_time", hue="name", ax=ax, style="name", dashes=False, markers=markers)
    ax.set_xlabel("Graph size")
    ax.set_ylabel("Exeuction time (ms)")
    time_min = df["avg_execution_time"].min()

    time_max = df["avg_execution_time"].max()
    if ceil_on_time_axis is not None:
        time_max = min(df["avg_execution_time"].max(), ceil_on_time_axis)

    ax.set_ylim((time_min - 0.05, time_max))
    yticks = [x / 10 for x in range(
        max(math.floor(time_min * 10 - 1), 0),
        math.ceil(time_max * 10 * 1.1))]
    ylabels = [f"{int(x * 1000)}" for x in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.legend()


def display_ablations(df: pandas.DataFrame, ax, colors=None, line_styles=None, fill_alpha=0.2):
    _unique_sizes = sorted([s for s in df["size"].unique()])
    ax.set_title("Ablation study")
    ax.set_xlabel("Graph size (number of nodes)")
    _x_min = 0
    _x_max = max(_unique_sizes) * 1.1
    ax.set_xlim(_x_min, _x_max)
    ax.set_xticks(_unique_sizes)
    ax.set_xticklabels(_unique_sizes)
    ax.set_ylabel("Fraction of graphs solved (HC found)")
    ax.set_ylim(-0.05, 1.1)
    _yticks = [0.1 * x for x in range(0, 11, 1)]
    ax.set_yticks(_yticks)
    ax.set_yticklabels([f"{x:.1f}" for x in _yticks])

    seaborn.lineplot(
        data=df, ax=ax, x="size", y="perc_hamilton_found", hue="class", errorbar=('sd', 2),
        style="class", markers=True, dashes=False)
    ax.legend()
