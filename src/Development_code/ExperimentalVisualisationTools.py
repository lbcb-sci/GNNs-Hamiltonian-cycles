import networkx as nx
import torch_geometric as torch_g
from matplotlib import pyplot as plt
from matplotlib import cm
from src.Models import HamiltonianCycleFinder
from src.VisualisationTools import display_result_on_known_hamilton_graphs


def display_decisions_step_by_step(d: torch_g.data.Data, tour, selection):
    d = d.clone()
    nx_g = torch_g.utils.to_networkx(d)
    layout = nx.spring_layout(nx_g)

    for step in range(0, len(tour) - 1):
        fig, ax = plt.subplots(1, 1)
        d.p = selection[step]

        nx_g = torch_g.utils.to_networkx(d, node_attrs=["p"])
        cmap = cm.get_cmap("Reds")

        nx.draw(nx_g, pos=layout,
                node_color=[(0., 0., 0., 1.) if n in tour[:step] else cmap(n_d["p"]) for n, n_d in nx_g.nodes.items()],
                with_labels=False,
                ax=ax)
        nx.draw_networkx_labels(nx_g, layout,
                                labels={n: "{:.0f}".format(n_d["p"] * 100) for n, n_d in nx_g.nodes.items()},
                                ax=ax)

        manager = plt.get_current_fig_manager()
        manager.window.state('zoomed')
        plt.show()
    plt.show()
    return tour


def visualize_nn_performance_on_saved_data(hamilton_nn: HamiltonianCycleFinder, data_directories):
    from src.DatasetBuilder import ErdosRenyiInMemoryDataset
    from src.Evaluation import EvaluationScores
    dataset = ErdosRenyiInMemoryDataset(data_directories)
    for d, hamilton_cylce in dataset:
        if len(hamilton_cylce) != d.num_nodes:
            continue
        if d.num_nodes < 30 or d.num_nodes > 50:
            continue
        # nn_path = hamilton_nn.run_greedy_neighbor(d)[0]
        b = torch_g.data.Batch.from_data_list([d])
        nn_path = hamilton_nn.batch_run_greedy_neighbor(b)[0]
        valid = EvaluationScores.verify_only_neighbor_connections(b, nn_path)
        print(valid)
        nn_path = [x.item() for x in nn_path.flatten()]

        # print(hamilton_cylce)
        # g = torch_g.utils.to_networkx(d)
        # nx.draw(g, with_labels=True)
        # plt.show()
        # continue

        start = nn_path[0]
        start_pos = hamilton_cylce.index(start)
        tmp_ham = hamilton_cylce[start_pos:] + hamilton_cylce[:start_pos]
        fig = display_result_on_known_hamilton_graphs(d, tmp_ham, tmp_ham)
        fig.show()
        plt.show()

        fig = display_result_on_known_hamilton_graphs(d, nn_path, hamilton_cylce)
        fig.show()
        plt.show()
    return
