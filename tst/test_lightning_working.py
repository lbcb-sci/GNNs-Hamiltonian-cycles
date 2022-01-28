import torch
from torch.utils.data import DataLoader
import pytorch_lightning as torch_lightning
from pytorch_lightning.loggers import WandbLogger
import wandb
import copy

import src.Models as Models
from src.data.InMemoryDataset import ErdosRenyiInMemoryDataset
from src.data.GraphGenerators import NoisyCycleGenerator
from src.data.GraphDataset import GraphDataLoader, GraphGeneratingDataset
from src.data.DataModules import ArtificialCycleDataModule


if __name__ == "__main__":
    torch.set_num_threads(32)
    wandb.init(project="gnns-Hamiltonian-cycles", mode="disabled")
    wandb_logger = WandbLogger()

    datamodule = ArtificialCycleDataModule()
    model = Models.EncodeProcessDecodeAlgorithm(is_load_weights=False, loss_type="entropy", processor_depth=5, hidden_dim=32)
    trainer = torch_lightning.Trainer(max_epochs=500, num_sanity_val_steps=2, check_val_every_n_epoch=2, logger=wandb_logger)

    trainer.fit(model=model, datamodule=datamodule)

    dataloader = datamodule.test_dataloader()
    if isinstance(dataloader.dataset, ErdosRenyiInMemoryDataset):
        sizes = set(graph_item.graph.num_nodes for graph_item in dataloader.dataset)
        test_dataloaders = []
        for size in sizes:
            subset_dataloader = copy.deepcopy(dataloader)
            subset_dataloader.dataset.data_list = \
                [data_item for data_item in subset_dataloader.dataset.data_list if data_item[ErdosRenyiInMemoryDataset.NUM_NODES_TAG] == size]
            test_dataloaders.append(subset_dataloader)
    else:
        test_dataloaders = [dataloader]

    trainer.test(model, dataloaders=test_dataloaders[1])
