import torch
from torch.utils.data import DataLoader
import pytorch_lightning as torch_lightning
from pytorch_lightning.loggers import WandbLogger
import wandb

import src.Models as Models
from src.data.InMemoryDataset import ErdosRenyiInMemoryDataset
from src.data.GraphGenerators import NoisyCycleGenerator
from src.data.GraphDataset import GraphDataLoader, GraphGeneratingDataset
from src.data.DataModules import ArtificialCycleDataModule


if __name__ == "__main__":
    generator = NoisyCycleGenerator(25, 0.8)
    dataset = GraphGeneratingDataset(generator)
    dataloader = GraphDataLoader(dataset, batch_size=10)
    model = Models.EncodeProcessDecodeAlgorithm(False, loss_type="entropy")
    wandb.init(project="gnns-Hamiltonian-cycles", mode=None)
    wandb_logger = WandbLogger()
    torch.set_num_threads(32)
    trainer = torch_lightning.Trainer(max_epochs=500, num_sanity_val_steps=2, check_val_every_n_epoch=2, logger=wandb_logger)
    datamodule = ArtificialCycleDataModule()
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model, test_dataloaders=datamodule.test_dataloader())
