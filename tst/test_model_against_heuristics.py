from sqlite3 import Timestamp
import time
import torch
from pathlib import Path
import pandas

from src.Development_code.Heuristics import LeastDegreeFirstHeuristics, HybridHam
from src.Evaluation import EvaluationPlots, EvaluationScores
from src.ExactSolvers import ConcordeHamiltonSolver

import src.model_utils as model_utils

DATA_CSV_PATH = Path(__file__).parent / "accuracy_scores.csv"

def test_models_against_heuristics():
    timestamp = time.time_ns()
    HamS_model, _ = model_utils.load_existing_model(model_identifier="s9ket5ka")
    # HamR_model = train_HamR(True, 0)

    hybrid_ham_heuristics = HybridHam()
    least_degree_first = LeastDegreeFirstHeuristics()
    concorde_solver = ConcordeHamiltonSolver()
    df = EvaluationScores.accuracy_scores_on_saved_data(
        [HamS_model, hybrid_ham_heuristics, least_degree_first, concorde_solver],
        ["HamS_model", "HybridHam", "Least_degree_first", "Concorde"], nr_graphs_per_size=10_000)
    df["timestamp"] = timestamp
    return df

if __name__ == '__main__':
    df_new = test_models_against_heuristics()
    if DATA_CSV_PATH.exists():
        df_old = pandas.read_csv(DATA_CSV_PATH)
        df = pandas.concat(df_old, df_new)
    else:
        df = df_new
    df.to_csv(DATA_CSV_PATH, index=False)