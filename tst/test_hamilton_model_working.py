import itertools
import itertools
import pickle
from pathlib import Path

import torch
import pytorch_lightning as torch_lightning
import torch_geometric as torch_g

from src.data.GraphGenerators import ErdosRenyiGenerator
from src.data.DataModules import ArtificialCycleDataModule
import src.Models as Models
from src.Evaluation import EvaluationScores

if __name__ == "__main__":
    weights_dir_path = Path("new_weights/")
    HamS = Models.EncodeProcessDecodeAlgorithm.load_from_checkpoint(weights_dir_path / "lightning_HamS.ckpt")
    HamS_datamodule = ArtificialCycleDataModule()
    HamS_trainer = torch_lightning.Trainer(max_epochs=0, num_sanity_val_steps=0)
    HamS_trainer.fit(HamS, datamodule=HamS_datamodule)

    graphs_path = Path("tst/small_graphs.pkl")
    if graphs_path.exists():
        with open(graphs_path, "rb") as in_file:
            graphs = pickle.load(in_file)
        print("loaded graphs")
    else:
        generator = ErdosRenyiGenerator(15, 0.8)
        graphs = [g for g in itertools.islice(generator, 10)]
        with open(graphs_path, "wb") as out:
            pickle.dump(graphs, out)

    old_walks = [HamS.batch_run_greedy_neighbor_old(torch_g.data.Batch.from_data_list([g])) for g in graphs]

    walks = HamS.solve_graphs(graphs)
    for g, w in zip(graphs, walks):
        print(f"Walk {w} is {'valid' if EvaluationScores.is_walk_valid(g, w) else 'invalid'}")
