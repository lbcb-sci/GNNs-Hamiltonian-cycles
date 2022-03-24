import copy

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytorch_lightning.callbacks import LearningRateMonitor

import src.Models as Models
import src.data.DataModules as DataModules
import src.model_utils as model_utils
import src.callbacks as my_callbacks

train_request_HamS_model = model_utils.ModelTrainRequest(
    model_class = Models.EncodeProcessDecodeAlgorithm,
    datamodule_class = DataModules.ArtificialCycleDataModule,
    model_checkpoint = None,
    model_hyperparams = {
        "processor_depth": 5,
        "loss_type": "entropy",
        "starting_learning_rate": 1e-4
    },
    trainer_hyperparams = {
        "max_epochs": 2000,
        "num_sanity_val_steps": 2,
        "log_every_n_steps": 5,
        "check_val_every_n_epoch": 5,
        "gradient_clip_algorithm": "norm",
    },
    datamodule_hyperparams = {
        "train_virtual_epoch_size": 8 * 100,
        "val_virtual_epoch_size": 8 * 50,
        "train_batch_size": 8,
        "val_batch_size": 8,
        "train_graph_size": 25,
        "train_expected_noise_edges_per_node": 3,
    },
    model_checkpoint_hyperparams = {
        "every_n_epochs": 1,
        "dirpath": None,
    }
)
train_request_HamS_model.arguments["trainer_hyperparams"]["callbacks"] = [LearningRateMonitor(logging_interval="step")] \
    + [my_callbacks.create_lp_callback(target_type, p_norm) for target_type
       in ["max_lp_logits", "max_lp_weights", "max_lp_gradients", "flat_lp_gradients"] for p_norm in [2.1, 1.2, 2, 2.2, 5, 15]]

train_request_HamS_quick = copy.deepcopy(train_request_HamS_model)
train_request_HamS_quick.arguments["trainer_hyperparams"].update({"max_epochs": 50})

train_request_HamS_mse_loss = copy.deepcopy(train_request_HamS_model)
train_request_HamS_mse_loss.arguments["model_hyperparams"].update({"loss_type": "mse"})


train_request_HamS_depth_3 = copy.deepcopy(train_request_HamS_model)
train_request_HamS_depth_3.arguments["model_hyperparams"].update({"processor_depth": 3})

train_request_HamS_depth_10 = copy.deepcopy(train_request_HamS_model)
train_request_HamS_depth_10.arguments["model_hyperparams"].update({"processor_depth": 10})

train_request_HamS_depth_15 = copy.deepcopy(train_request_HamS_model)
train_request_HamS_depth_15.arguments["model_hyperparams"].update({"processor_depth": 15})


train_request_HamS_graphs_50 = copy.deepcopy(train_request_HamS_model)
train_request_HamS_graphs_50.arguments["datamodule_hyperparams"].update({"train_graph_size": 50, "val_graph_size": 50})

train_request_HamS_graphs_100 = copy.deepcopy(train_request_HamS_model)
train_request_HamS_graphs_100.arguments["datamodule_hyperparams"].update({"train_graph_size": 50, "val_graph_size": 30})


train_request_HamS_ER_exact_solver = copy.deepcopy(train_request_HamS_model)
train_request_HamS_ER_exact_solver.arguments["datamodule_class"] = DataModules.SolvedErdosRenyiDataModule
train_request_HamS_ER_exact_solver.arguments["datamodule_hyperparams"].update({"train_hamilton_existence_probability": 0.8})
del train_request_HamS_ER_exact_solver.arguments["datamodule_hyperparams"]["train_expected_noise_edges_per_node"]

train_request_HamS_grad_clipping = copy.deepcopy(train_request_HamS_model)
train_request_HamS_grad_clipping.arguments["trainer_hyperparams"]["gradient_clip_val"] = 1

train_request_HamS_custom_lr = copy.deepcopy(train_request_HamS_model)
train_request_HamS_custom_lr.arguments["model_hyperparams"]["starting_learning_rate"] = 1e-5

train_request_HamS_automatic_lr = copy.deepcopy(train_request_HamS_grad_clipping)
train_request_HamS_automatic_lr.arguments["trainer_hyperparams"]["auto_lr_find"] = True
train_request_HamS_automatic_lr.arguments["trainer_hyperparams"]["track_grad_norm"] = 2


train_request_HamS_cosine_annealing = copy.deepcopy(train_request_HamS_grad_clipping)
train_request_HamS_cosine_annealing.arguments["model_hyperparams"]["lr_scheduler_class"] = CosineAnnealingWarmRestarts
train_request_HamS_cosine_annealing.arguments["model_hyperparams"]["starting_learning_rate"] = 2*1e-4
train_request_HamS_cosine_annealing.arguments["model_hyperparams"]["lr_scheduler_hyperparams"] = {
    "T_0": 400,
    "eta_min": 1e-6,
}

_debug_large_lr = copy.deepcopy(train_request_HamS_model)
_debug_large_lr.arguments["model_hyperparams"]["starting_learning_rate"] = 0.012
# _debug_large_lr.arguments["trainer_hyperparams"]["track_grad_norm"] = 2

_debug_with_multiple_dataloaders = copy.deepcopy(train_request_HamS_model)
_debug_with_multiple_dataloaders.arguments["model_hyperparams"]["val_dataloader_tags"] = ["artificial", "ER"]
_debug_with_multiple_dataloaders.arguments["datamodule_class"] = DataModules.ArtificialCycleWithDoubleEvaluationDataModule
_debug_with_multiple_dataloaders.arguments["datamodule_hyperparams"].update({
    "val_hamiltonian_existence_probability": 0.8,
    "val_virtual_epoch_size": 80,
})
