import copy

import src.Models as Models
import src.data.DataModules as DataModules
import src.model_utils as model_utils

train_request_HamS_model = model_utils.ModelTrainRequest(
    model_class = Models.EncodeProcessDecodeAlgorithm,
    datamodule_class = DataModules.ArtificialCycleDataModule,
    model_checkpoint = None,
    model_hyperparams = {
        "processor_depth":5,
        "loss_type": "entropy",
    },
    trainer_hyperparams = {
        "max_epochs": 2,
        "num_sanity_val_steps": 2,
        "log_every_n_steps": 5,
        "check_val_every_n_epoch": 5,
    },
    datamodule_hyperparams = {
        "train_virtual_epoch_size": 32 * 100,
        "val_virtual_epoch_size": 32 * 50,
        "train_batch_size": 32,
        "val_batch_size": 32,
        "train_graph_size": 25,
        "train_noise_prob_per_edge": 1/25,
    },
    model_checkpoint_hyperparams = {
        "every_n_epochs": 1,
        "dirpath": None,
    }
)

train_request_HamS_mse_loss = copy.deepcopy(train_request_HamS_model)
train_request_HamS_mse_loss.arguments["model_hyperparams"].update({"loss_type": "mse"})

train_request_HamS_expanded = copy.deepcopy(train_request_HamS_model)
train_request_HamS_expanded.arguments["model_hyperparams"].update({"processor_depth": 10})

train_request_HamS_larger_graphs = copy.deepcopy(train_request_HamS_model)
train_request_HamS_larger_graphs.arguments["datamodule_hyperparams"].update({"train_graph_size": 50, "val_graph_size": 50})
