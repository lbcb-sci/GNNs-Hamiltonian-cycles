import os

from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as torch_lightning
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning.callbacks
import torch_geometric
import wandb

from src.Evaluation import EvaluationScores
import src.Models as Models
from src.data.InMemoryDataset import ErdosRenyiInMemoryDataset
from src.data.GraphGenerators import ErdosRenyiGenerator, NoisyCycleGenerator
from src.data.GraphDataset import GraphDataLoader, GraphGeneratingDataset
from src.data.DataModules import ArtificialCycleDataModule, ErdosRenyiInMemoryDataset, ReinforcementErdosRenyiDataModule


def check_existing_models_working(wandb_project=None, weights_dir_path=Path("new_weights"), is_wandb_log_offline=True):
    HamR_path = weights_dir_path / "lightning_HamR.ckpt"
    HamS_path = weights_dir_path / "lightning_HamS.ckpt"
    is_wandb_log_offline = True
    if wandb_project is None:
        wandb_project = "Unnamed_project"
        is_wandb_log_offline = True

    print(f"Testing performance HamR and HamS models from {HamR_path} and {HamS_path}")

    HamR = Models.GatedGCNEmbedAndProcess.load_from_checkpoint(HamR_path)
    HamR_datamodule = ReinforcementErdosRenyiDataModule(HamR)
    HamR_trainer = torch_lightning.Trainer(max_epochs=0, num_sanity_val_steps=0, logger=WandbLogger(project=wandb_project, offline=is_wandb_log_offline))
    HamR_trainer.fit(HamR, datamodule=HamR_datamodule)
    test_dataloaders = {int(path.split('(')[1].split(',')[0]): GraphDataLoader(ErdosRenyiInMemoryDataset([os.path.join("DATA", path)]), 1) for path in os.listdir("DATA") if path.endswith(".pt")}
    _original_log_test_tag = HamR.log_test_tag
    HamR_test_results = {}
    for graph_size, dataloader in test_dataloaders.items():
        if graph_size > 100:
            continue
        HamR.log_test_tag = f"test_graph_size_{graph_size}"
        HamR_test_results.update({graph_size: HamR_trainer.test(HamR, dataloaders=[dataloader], verbose=False)})
    HamR.log_test_tag =  _original_log_test_tag
    wandb.finish()
    print(HamR_test_results)

    HamS = Models.EncodeProcessDecodeAlgorithm.load_from_checkpoint(HamS_path)
    HamS_datamodule = ArtificialCycleDataModule()
    HamS_trainer = torch_lightning.Trainer(max_epochs=0, num_sanity_val_steps=0, logger=WandbLogger(project=wandb_project, offline=is_wandb_log_offline))
    HamS_trainer.fit(HamS, datamodule=HamS_datamodule)
    _original_log_test_tag = HamS.log_test_tag
    HamS_test_results = {}
    for graph_size, dataloader in test_dataloaders.items():
        if graph_size > 100:
            continue
        HamS.log_test_tag = f"test_graph_size_{graph_size}"
        HamS_test_results.update({graph_size: HamS_trainer.test(HamS, dataloaders=[dataloader], verbose=False)})
    HamS.log_test_tag = _original_log_test_tag
    wandb.finish()
    print(HamS_test_results)

    print(f"Success, HamR and HamS models from {weights_dir_path} tests working")


if __name__ == "__main__":
    torch.set_num_threads(32)
    wandb_project = "gnns-Hamiltonian-cycles"

    check_existing_models_working(wandb_project=wandb_project)
