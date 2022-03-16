import wandb

import src.constants as constants
from src.Development_code.Heuristics import LeastDegreeFirstHeuristics, HybridHam
from src.ExactSolvers import ConcordeHamiltonSolver
import src.model_utils as model_utils

def test_heuristics_and_add_to_wandb(wandb_project=constants.WEIGHTS_AND_BIASES_PROJECT):
    heuristcs_data = [
        (LeastDegreeFirstHeuristics(), "LeastDegreeFirstHeuristics"),
        (HybridHam(), "HybridHamHeuristics"),
        (ConcordeHamiltonSolver(), "ConcordeSolver")
    ]
    for solver, name in heuristcs_data:
        wandb_run = wandb.init(job_type="test", project=wandb_project, tags=["benchmark"], name=name)
        model_utils.test_on_saved_data(solver, wandb_run)
        wandb_run.finish()


if __name__ == "__main__":
    test_heuristics_and_add_to_wandb()
