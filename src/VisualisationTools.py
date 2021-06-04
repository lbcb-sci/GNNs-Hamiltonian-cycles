import networkx as nx
import torch_geometric as torch_g
from matplotlib import pyplot as plt


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


def display_ER_graph(d: torch_g.data):
    g = torch_g.utils.to_networkx(d)
    g = nx.to_undirected(g)
    fig = plt.figure(figsize=DEFAULT_FIG_SIZE)
    ax = fig.add_subplot(1, 1, 1)
    nx.draw(g, nx.spring_layout(g), ax=ax)
    return fig

