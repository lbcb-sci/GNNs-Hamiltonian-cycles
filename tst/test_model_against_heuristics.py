from operator import index
from src.Development_code.Heuristics import LeastDegreeFirstHeuristics, HybridHam
from src.Evaluation import EvaluationPlots, EvaluationScores
from train import train_HamR, train_HamS
import time
import torch
from pathlib import Path
import pandas

DATA_CSV_PATH = Path(__file__).parent / "accuracy_scores.csv"

def test_models_against_heuristics():
    torch.set_num_threads(1)
    timestamp = time.time()
    HamS_model = train_HamS(True, 0)
    HamR_model = train_HamR(True, 0)
    hybrid_ham_heuristics = HybridHam()
    least_degree_first = LeastDegreeFirstHeuristics()
    df = EvaluationScores.accuracy_scores_on_saved_data(
        [HamS_model, HamR_model, hybrid_ham_heuristics, least_degree_first],
        ["HamS", "HamR", "HybridHam", "Least_degree_first"], nr_graphs_per_size=10_000)
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