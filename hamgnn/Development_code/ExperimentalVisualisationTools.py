import networkx as nx
import torch_geometric as torch_g
from matplotlib import pyplot as plt
from matplotlib import cm
from typing import List
import pandas
import seaborn
from matplotlib import pyplot

from src.Models import HamFinderGNN, HamiltonSolver
from src.VisualisationTools import display_result_on_known_hamilton_graphs
from src.Evaluation import EvaluationScores, EvaluationPlots
from src.Development_code.Heuristics import HybridHam, LeastDegreeFirstHeuristics


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