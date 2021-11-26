import networkx as nx
import torch_geometric as torch_g
from matplotlib import pyplot as plt
from matplotlib import cm
from typing import List
import pandas
import seaborn
from matplotlib import pyplot

from src.Models import HamFinderGNN, HamFinder
from src.VisualisationTools import display_result_on_known_hamilton_graphs
from src.Evaluation import EvaluationScores
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


def visualize_nn_performance_on_saved_data(hamilton_solvers: List[HamFinder], solver_names, data_dir=None):
    scores_list = []
    for solver, name in zip(hamilton_solvers, solver_names):
        evals = EvaluationScores.evaluate_on_saved_data(solver.solve_graphs, nr_graphs_per_size=None, data_folders=data_dir)
        scores = EvaluationScores.compute_accuracy_scores(evals)
        scores["solver_name"] = name
        scores_list.append(scores)
    df_scores = pandas.concat(scores_list).reset_index(drop=True)
    combined = []
    for accuracy_col in [c for c in df_scores.columns if c.startswith("perc")]:
        partial_df = df_scores[["size", "solver_name", accuracy_col]].copy()
        partial_df.columns = [c if c != accuracy_col else "perc_score" for c in partial_df.columns]
        partial_df["score_type"] = accuracy_col
        combined.append(partial_df)
    
    df_scores = pandas.concat(combined, ignore_index=True)
    seaborn.FacetGrid(data=df_scores, col="score_type").map_dataframe(seaborn.lineplot, x="size", y="perc_score", hue="solver_name")
    plt.show()


if __name__ == '__main__':
    from train import train_HamS
    HamS_model = train_HamS(True, 0)
    hybrid_ham_heuristics = HybridHam()
    least_degree_first = LeastDegreeFirstHeuristics()
    visualize_nn_performance_on_saved_data(
        [HamS_model, hybrid_ham_heuristics, least_degree_first],
        ["HamS", "HybridHam", "Least_degree_first"], )
