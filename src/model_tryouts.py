import torch
import copy
import sys

import src.Models as Models
import src.data.DataModules as DataModules
import src.constants as constants
import src.train_model as train_model


class ModelTrainRequest:
    def __init__(self, **kwargs):
        self.arguments = kwargs

    def train(self, nr_cpu_threads=16):
        torch.set_num_threads(nr_cpu_threads)
        train_model.train_model(**self.arguments)


train_request_HamS_model = ModelTrainRequest(
    model_class = Models.EncodeProcessDecodeAlgorithm,
    datamodule_class = DataModules.ArtificialCycleDataModule,
    model_checkpoint = None,
    model_hyperparams = {
        "processor_depth":5,
        "loss_type": "entropy",
    },
    trainer_hyperparams = {
        "max_epochs": 200,
        "num_sanity_val_steps": 2,
        "log_every_n_steps": 5,
        "check_val_every_n_epoch": 5
    },
    datamodule_hyperparams = {
        "train_virtual_epoch_size": 32 * 100,
        "val_virtual_epoch_size": 32 * 50,
        "train_batch_size": 32,
        "val_batch_size": 32,
        "train_graph_size": 25,
        "train_noise_prob_per_edge": 1/25,
    }
)

train_request_HamS_mse_loss = copy.deepcopy(train_request_HamS_model)
train_request_HamS_mse_loss.arguments["model_hyperparams"].update({"loss_type": "mse"})

train_request_HamS_expanded = copy.deepcopy(train_request_HamS_model)
train_request_HamS_expanded.arguments["model_hyperparams"].update({"processor_depth": 10})

train_request_HamS_larger_graphs = copy.deepcopy(train_request_HamS_model)
train_request_HamS_larger_graphs.arguments["datamodule_hyperparams"].update({"train_graph_size": 50, "val_graph_size": 50})


if __name__ == "__main__":
    possible_targets = [var_name for var_name, var in locals().items() if isinstance(var, ModelTrainRequest)]
    args = sys.argv
    assert len(args) <= 2
    if len(args) == 2:
        target_name = args[-1]
    else:
        target_name = "train_request_HamS_larger_graphs"

    if target_name in possible_targets:
        locals()[target_name].train()
    else:
        print(f"Could not find model request {target_name} to train. Please use one of the following as an argument")
        for possible_target in possible_targets:
            print(f'"{possible_target}"\n')