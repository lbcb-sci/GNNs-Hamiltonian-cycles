import itertools
from train import train_HamS
from src.GraphGenerators import ErdosRenyiGenerator
import itertools
from src.Evaluation import EvaluationScores
import torch_geometric as torch_g
import pickle
from pathlib import Path

if __name__ == "__main__":
    model = train_HamS(True, 0)
    graphs_path = Path("tests/small_graphs.pkl")
    if graphs_path.exists():
        with open(graphs_path, "rb") as in_file:
            graphs = pickle.load(in_file)
        print("loaded graphs")
    else:
        generator = ErdosRenyiGenerator(12, 0.8)
        graphs = [g for g in itertools.islice(generator, 4)]
        with open(graphs_path, "wb") as out:
            pickle.dump(graphs, out)

    walks = model.solve_graphs(graphs)
    for g, w in zip(graphs, walks):
        print(f"Walk {w} is {'valid' if EvaluationScores.is_walk_valid(g, w) else 'invalid'}")
