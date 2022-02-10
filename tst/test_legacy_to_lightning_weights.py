import os
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as torch_lightning
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning.callbacks

import src.legacy.legacy_model_weights_io as legacy_io
from src.Evaluation import EvaluationScores
import src.Models as Models
from src.data.InMemoryDataset import ErdosRenyiInMemoryDataset
from src.data.GraphGenerators import NoisyCycleGenerator
from src.data.GraphDataset import GraphDataLoader, GraphGeneratingDataset
from src.data.DataModules import ArtificialCycleDataModule, ErdosRenyiInMemoryDataset, ReinforcementErdosRenyiDataModule

if __name__ == "__main__":
    new_weights_directory = Path("tst/test_legacy_to_lightning_weights")

    HamS = legacy_io.load_legacy_HamS()
    datamodule = ArtificialCycleDataModule()
    trainer = pytorch_lightning.Trainer(max_epochs=0, num_sanity_val_steps=0)
    trainer.fit(HamS, datamodule=datamodule)
    trainer.save_checkpoint(new_weights_directory / "lightning_HamS.ckpt")
    lightning_HamS = Models.EncodeProcessDecodeAlgorithm.load_from_checkpoint(new_weights_directory / "lightning_HamS.ckpt")

    HamR = legacy_io.load_legacy_HamR()
    trainer = pytorch_lightning.Trainer(max_epochs=0, num_sanity_val_steps=0)
    trainer.fit(HamR, datamodule=datamodule)
    trainer.save_checkpoint(new_weights_directory / "lightning_HamR.ckpt")
    lightning_HamR = Models.GatedGCNEmbedAndProcess.load_from_checkpoint(new_weights_directory / "lightning_HamR.ckpt")

    print("Checking that new models have the same parameters...")
    for old_parameter, new_parameter in tqdm(zip(HamS.parameters(), lightning_HamS.parameters())):
        check = torch.isclose(old_parameter, new_parameter).all().item()
        assert check is True
    for old_parameter, new_parameter in zip(HamR.parameters(), lightning_HamR.parameters()):
        assert torch.isclose(old_parameter, new_parameter).all()
    print("Everything seems to be in order")
