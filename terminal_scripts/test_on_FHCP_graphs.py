from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import pandas
import wandb
import torch

from hamgnn.data.FHCPDataset import FHCPDataset
from hamgnn.heuristics import HybridHam, LeastDegreeFirstHeuristics
from hamgnn.ExactSolvers import ConcordeHamiltonSolver
import hamgnn.constants as constants
from hamgnn.Evaluation import EvaluationScores
import hamgnn.model_utils as model_utils
import hamgnn.constants as constants
from terminal_scripts.test_heuristics import HeuristicsNames



if __name__ == "__main__":
    parser = ArgumentParser(f"Tests performance of model on FHCP dataset from {constants.FHCP_HOMEPAGE}")
    _heuristic_string = " ".join([f'{s}' for s in [HeuristicsNames.CONCORDE, HeuristicsNames.HYBRID_HAM, HeuristicsNames.LEAST_DEGREE_FIRST]])
    parser.add_argument("identifier", help=f"Either a w&b id of run or one of {_heuristic_string} to indicate heuristic")
    parser.add_argument("out", help=f"Output .csv file to write the results to")
    args = parser.parse_args()
    out_path = Path(args.out)
    if out_path.exists():
        print(f"Output file {out_path} already_exists. Aborting")
        exit(0)
    identifier = args.identifier
    if identifier == HeuristicsNames.HYBRID_HAM:
        model = HybridHam()
    elif identifier == HeuristicsNames.LEAST_DEGREE_FIRST:
        model = LeastDegreeFirstHeuristics()
    elif identifier == HeuristicsNames.CONCORDE:
        model = ConcordeHamiltonSolver()
    else:
        try:
            wandb_project = constants.WEIGHTS_AND_BIASES_PROJECT
            wandb_id = identifier
            wandb_run = wandb.init(project=wandb_project, id=wandb_id, resume=True)
            model = model_utils.create_model_for_wandb_run(wandb_run, wandb_run.config["checkpoint"])
            if torch.cuda.is_available():
                model = model.cuda()
        except Exception as ex:
            print(f"Could not load model from identifier {identifier}. Aborting.\nDetailed error message {ex}")
            exit(-1)

    fhcp_dataset = FHCPDataset(constants.FHCP_BENCHMARK_DIR)
    graph_list = list(tqdm((ex.graph for ex in fhcp_dataset), desc="Loading FHCP graphs...", total=len(fhcp_dataset)))
    metadata_dict = defaultdict(list)
    for index in range(len(fhcp_dataset)):
        filename, number = fhcp_dataset.get_problem_description(index)
        metadata_dict["filename"].append(filename)
        metadata_dict["number"].append(number)
    evals = EvaluationScores().solve_time_and_evaluate(model.timed_solve_graphs, graph_list, is_show_progress=True)
    evals.update(metadata_dict)
    df_evals = pandas.DataFrame(evals)
    df_evals.to_csv(out_path)
