import copy

import torch
from pytorch_lightning import Trainer
import pytorch_lightning

import src.constants as constants
import src.Models as Models
import src.data.GraphDataset as GraphDataset
from src.data.InMemoryDataset import ErdosRenyiInMemoryDataset
import src.model_utils as model_utils

if __name__ == "__main__":
    args =
    wandb_kwargs = {"project": wandb_project, "resume": True}
    if wandb_id is not None:
        wandb_kwargs["id"] = wandb_id
    wandb.init(**wandb_kwargs)
    model_class = Models.EncodeProcessDecodeAlgorithm
    model_checkpoint = None
    model = Models.EncodeProcessDecodeAlgorithm()
    results = model_utils.test_on_saved_data(model)

    # model = model_class.load_from_checkpoint(model_checkpoint)

    exit(-1)
    results = model_utils.test_model_on_saved_data(model_class, )
    print(results)
