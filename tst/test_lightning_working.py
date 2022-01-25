import torch
from torch.utils.data import DataLoader
import pytorch_lightning as torch_lightning

import src.Models as Models
from src.data.InMemoryDataset import ErdosRenyiInMemoryDataset
from src.data.GraphGenerators import NoisyCycleGenerator
from src.data.GraphDataset import GraphDataLoader, GraphGeneratingDataset
from src.data.DataModules import ArtificialCycleDataModule


if __name__ == "__main__":
    generator = NoisyCycleGenerator(100, 0.8)
    dataset = GraphGeneratingDataset(generator)
    dataloader = GraphDataLoader(dataset, batch_size=10)
    model = Models.EncodeProcessDecodeAlgorithm(False)
    torch.set_num_threads(16)
    trainer = torch_lightning.Trainer(max_epochs=20, num_sanity_val_steps=2, val_check_interval=10)
    datamodule = ArtificialCycleDataModule()
    trainer.fit(model=model, datamodule=datamodule)
