from src.Development_code.Heuristics import LeastDegreeFirstHeuristics, HybridHam
from src.Evaluation import EvaluationPlots

if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    from train import train_HamS
    HamS_model = train_HamS(False, 0)
    hybrid_ham_heuristics = HybridHam()
    least_degree_first = LeastDegreeFirstHeuristics()
    fig = EvaluationPlots.accuracy_curves_for_saved_data(
        [HamS_model, hybrid_ham_heuristics, least_degree_first],
        ["HamS", "HybridHam", "Least_degree_first"], nr_graphs_per_size=10_000)
    fig.show()