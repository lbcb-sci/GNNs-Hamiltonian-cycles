import sys
import wandb

import src.constants as constants
from src.Development_code.Heuristics import LeastDegreeFirstHeuristics, HybridHam
from src.ExactSolvers import ConcordeHamiltonSolver
import src.model_utils as model_utils

class HeuristicsNames:
    CONCORDE = "concorde"
    HYBRID_HAM = "hybrid_ham"
    LEAST_DEGREE_FIRST = "least_degree_first"
    ALL_HEURISTICS = [CONCORDE, HYBRID_HAM, LEAST_DEGREE_FIRST]


def evaluate_heuristic(wandb_project=constants.WEIGHTS_AND_BIASES_PROJECT, heuristics: list[str] = None, wandb_ids: list[str] = None):
    if heuristics is None:
        heuristics = [HeuristicsNames.CONCORDE, HeuristicsNames.HYBRID_HAM, HeuristicsNames.LEAST_DEGREE_FIRST]
        wandb_ids = None
    if wandb_ids is None:
        wandb_ids = [None for h in heuristics]
    assert len(heuristics) == len(wandb_ids), "Please provide wandb_ids for each heuristics to update"
    heuristics_data = {
        HeuristicsNames.LEAST_DEGREE_FIRST: (LeastDegreeFirstHeuristics(), "LeastDegreeFirstHeuristics"),
        HeuristicsNames.HYBRID_HAM: (HybridHam(), "HybridHamHeuristics"),
        HeuristicsNames.CONCORDE: (ConcordeHamiltonSolver(), "ConcordeSolver")
    }
    for h in heuristics:
        assert h in heuristics_data, f"parameter {h} needs to be one of {heuristics_data.keys()}"

    for heuristic, wandb_id in zip(heuristics, wandb_ids):
        solver, name = heuristics_data[heuristic]
        if wandb_id is None:
            wandb_run = wandb.init(job_type="test", project=wandb_project, tags=["benchmark"], name=name)
        else:
            wandb_run = wandb.init(project=wandb_project, id=wandb_id, resume=True)
        model_utils.test_on_saved_data(solver, wandb_run)
        wandb_run.finish()


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2 or len(args) > 3:
        print("Please provide the name of the heuristic to test with or without wandb_id of the run to append it to")
        exit(-115)
    elif len(args) == 2:
        heuristics = [args[1]]
        wandb_ids = [None]
    elif len(args) == 3:
        heuristics = [args[1]]
        wandb_ids = [args[2]]
    evaluate_heuristic(heuristics=heuristics, wandb_ids=wandb_ids)
