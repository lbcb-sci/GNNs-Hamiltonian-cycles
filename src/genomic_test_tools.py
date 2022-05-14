from pathlib import Path
import pandas

import wandb

from OlcGraph import OlcGraph

import src.model_utils as model_utils
from src.HamiltonSolver import HamiltonSolver
import src.constants as constants
from src.Evaluation import EvaluationScores

def test_on_genomic_graphs(model: HamiltonSolver, graph_paths: list[Path]):
    assert all([path.name.endswith(".gml") for path in graph_paths])
    graphs = [OlcGraph.from_gml(path) for path in graph_paths]
    solutions = model.solve_graphs(graphs)
    eval = EvaluationScores.evaluate(graphs, solutions)

    eval["graph_size"] = [graph.num_nodes for graph in graphs]
    eval["nr_incorrect_edges"] = []
    for graph, solution in zip(graphs, solutions):
        nx_g = graph.to_networkx()
        nr_wrong_edges = 0
        for i in range(len(solution) - 1):
            if not nx_g.edges[solution[i], solution[i + 1]][OlcGraph.ASSEMBLY_TAG_OVERLAP_REFERENCE_CORRECT]:
                nr_wrong_edges += 1
        eval["nr_incorrect_edges"].append(nr_wrong_edges)
    return eval


def compute_genomics_statistic(eval: dict):
    df = pandas.DataFrame(eval)
    df["coverage"] = df["length"] / df["graph_size"]
    return {
        "dataset_size": len(df.index),
        "graph_coverage": df["coverage"].mean(),
        "perc_cycles": df["is_cycle"].mean(),
        "perc_ham_cycles": (df["is_cycle"] & (df["graph_size"] == df["length"])).mean(),
        "cycles_graph_coverage": df[df["is_cycle"]]["coverage"].mean(),
        "perc_incorrect": (df["nr_incorrect_edges"] > 0).mean(),
        "avg_incorrect_edges": df["nr_incorrect_edges"].mean()
    }


def test_on_genomic_dataset(model: HamiltonSolver, dataset_summary_file=None):
    graph_paths = [Path(p) for p in dataset_summary_file.read_text().split("\n") if p.endswith(".gml")]
    eval = test_on_genomic_graphs(model, graph_paths)
    stats = compute_genomics_statistic(eval)
    return stats, eval

def test_wandb_model_on_genomic_data(wandb_id: str, model:HamiltonSolver=None, dataset_summary_file=None, log_prefix="string_graphs", wandb_project=constants.WEIGHTS_AND_BIASES_PROJECT):
    stats, eval = None, None
    try:
        wandb_run = wandb.init(project=wandb_project, id=wandb_id, resume=True)
        if model is None:
            model = model_utils.create_model_for_wandb_run(wandb_run)
        stats, eval = test_on_genomic_dataset(model, dataset_summary_file)
        wandb_run.log({f"{log_prefix}/{key}": value for key, value in stats.items()})
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    return stats, eval
