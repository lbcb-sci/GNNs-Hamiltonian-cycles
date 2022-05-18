from pathlib import Path
import wandb


from src.ExactSolvers import ConcordeHamiltonSolver
from src.Development_code.Heuristics import HybridHam
from src.genomic_test_tools import *


if __name__ == "__main__":
    concorde_solver = ConcordeHamiltonSolver()
    hybrid_ham = HybridHam()

    datasets_root = Path(__file__).parent.parent / "genome_graphs/SnakemakePipeline/"
    old_dataset = datasets_root / "analysis_genome_dataset.txt"
    clean_dataset = datasets_root / "correct_and_interesting_string_graphs.txt"
    failed_dataset = datasets_root / "failed_string_graphs.txt"
    full_dataset = datasets_root / "all_string_graphs.txt"

    hybrid_ham_run_id = "3ehji1qz"
    concorde_run_id = "ppkzdrb3"
    hams_run_id = "s9ket5ka"

    run_id = concorde_run_id

    # wandb_run = model_utils.reconnect_to_wandb_run(run_id)
    # hams = model_utils.create_model_for_wandb_run(wandb_run)
    # wandb_run.finish()

    model = concorde_solver

    stats, eval = test_wandb_model_on_genomic_data(run_id, model, dataset_summary_file=full_dataset, log_prefix="all_string_graphs")
    for key, value in stats.items():
        print(f"{key}", value)
