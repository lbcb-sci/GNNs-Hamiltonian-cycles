import torch
from torch.utils.data import DataLoader
import pytorch_lightning as torch_lightning
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning.callbacks
import wandb
from pathlib import Path

import src.Models as Models
from src.data.InMemoryDataset import ErdosRenyiInMemoryDataset
from src.data.GraphGenerators import NoisyCycleGenerator
from src.data.GraphDataset import GraphDataLoader, GraphGeneratingDataset
from src.data.DataModules import ArtificialCycleDataModule, ErdosRenyiInMemoryDataset, ReinforcementErdosRenyiDataModule


def check_existing_models_training(weights_dir_path=Path("new_weights")):
    torch.set_num_threads(8)

    # HamS = Models.EncodeProcessDecodeAlgorithm.load_from_checkpoint(weights_dir_path / "lightning_HamS.ckpt")
    # HamS_datamodule = ArtificialCycleDataModule()
    # HamS_trainer = torch_lightning.Trainer(max_epochs=2, num_sanity_val_steps=2)
    # HamS_trainer.fit(HamS, datamodule=HamS_datamodule)

    HamR = Models.GatedGCNEmbedAndProcess.load_from_checkpoint(weights_dir_path / "lightning_HamR.ckpt")
    HamR_datamodule = ReinforcementErdosRenyiDataModule(HamR)
    HamR_trainer = torch_lightning.Trainer(max_epochs=10, num_sanity_val_steps=2)
    HamR_trainer.fit(HamR, datamodule=HamR_datamodule)


if __name__ == "__main__":
    torch.set_num_threads(32)
    check_existing_models_training()
    exit()

    wandb.init(project="gnns-Hamiltonian-cycles", mode="disabled")
    wandb_logger = WandbLogger()
    checkpoint_saving_dir = "checkpoints"

    datamodule = ArtificialCycleDataModule()

    model = Models.EncodeProcessDecodeAlgorithm(is_load_weights=False, loss_type="entropy", processor_depth=5, hidden_dim=32)
    checkpoint_callback = torch_lightning.callbacks.ModelCheckpoint(monitor="validation/loss", dirpath=checkpoint_saving_dir)
    trainer = torch_lightning.Trainer(max_epochs=2, num_sanity_val_steps=2, check_val_every_n_epoch=2, callbacks=[checkpoint_callback], logger=wandb_logger)

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

    best_model_path = checkpoint_callback.best_model_path
    print(best_model_path)
    model = Models.EncodeProcessDecodeAlgorithm.load_from_checkpoint(best_model_path)

    trainer.test(model, dataloaders=test_dataloaders[1])
