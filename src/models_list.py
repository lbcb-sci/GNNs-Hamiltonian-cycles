import copy

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import src.Models as Models
import src.data.DataModules as DataModules
import src.model_utils as model_utils
import src.callbacks as my_callbacks

train_request_HamS_model = model_utils.ModelTrainRequest(
    model_class = Models.EncodeProcessDecodeAlgorithm,
    datamodule_class = DataModules.ArtificialCycleWithDoubleEvaluationDataModule,
    model_checkpoint = None,
    model_hyperparams = {
        "processor_depth": 5,
        "loss_type": "entropy",
        "starting_learning_rate": 1e-4,
        "val_dataloader_tags": ["artificial", "ER"],
        "starting_learning_rate": 4*1e-4,
        "lr_scheduler_class": CosineAnnealingWarmRestarts,
        "lr_scheduler_hyperparams": {
            "T_0": 400,
            "eta_min": 1e-6
        }
    },
    trainer_hyperparams = {
        "max_epochs": 2000,
        "num_sanity_val_steps": 2,
        "log_every_n_steps": 5,
        "check_val_every_n_epoch": 5,
        "gradient_clip_algorithm": "norm",
        "gradient_clip_val": 1
    },
    datamodule_hyperparams = {
        "train_virtual_epoch_size": 8 * 100,
        "val_virtual_epoch_size": 8 * 50,
        "train_batch_size": 8,
        "val_batch_size": 8,
        "train_graph_size": 25,
        "train_expected_noise_edges_per_node": 3,
        "val_hamiltonian_existence_probability": 0.8,
    },
    model_checkpoint_hyperparams = {
        "every_n_epochs": 1,
        "dirpath": None,
    }
)
lr_callbacks = [LearningRateMonitor(logging_interval="step")]
norm_monitoring_callbacks = [my_callbacks.create_lp_callback(target_type, p_norm) for target_type
       in ["max_lp_logits", "max_lp_weights", "max_lp_gradients", "flat_lp_gradients"] for p_norm in [2, 5]]
checkpoint_callbacks = [ModelCheckpoint(save_top_k=3, save_last=True, monitor="val/ER/hamiltonian_cycle")]
train_request_HamS_model.arguments["trainer_hyperparams"]["callbacks"] = lr_callbacks + norm_monitoring_callbacks + checkpoint_callbacks


train_request_HamS_quick = copy.deepcopy(train_request_HamS_model)
train_request_HamS_quick.arguments["trainer_hyperparams"].update({"max_epochs": 50})

train_request_HamS_mse_loss = copy.deepcopy(train_request_HamS_model)
train_request_HamS_mse_loss.arguments["model_hyperparams"].update({"loss_type": "mse"})


train_request_HamS_depth_3 = copy.deepcopy(train_request_HamS_model)
train_request_HamS_depth_3.arguments["model_hyperparams"].update({"processor_depth": 3})

train_request_HamS_depth_7 = copy.deepcopy(train_request_HamS_model)
train_request_HamS_depth_7.arguments["model_hyperparams"].update({"processor_depth": 7})

train_request_HamS_depth_15 = copy.deepcopy(train_request_HamS_model)
train_request_HamS_depth_15.arguments["model_hyperparams"].update({"processor_depth": 15})


train_request_HamS_graphs_50 = copy.deepcopy(train_request_HamS_model)
train_request_HamS_graphs_50.arguments["datamodule_hyperparams"].update({"train_graph_size": 50, "val_graph_size": 50})

train_request_HamS_graphs_100 = copy.deepcopy(train_request_HamS_model)
train_request_HamS_graphs_100.arguments["datamodule_hyperparams"].update({"train_graph_size": 100, "val_graph_size": 100})

train_request_HamS_graphs_200 = copy.deepcopy(train_request_HamS_model)
train_request_HamS_graphs_200.arguments["datamodule_hyperparams"].update({"train_graph_size": 200, "val_graph_size": 200})


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

train_request_HamS_different_artificial_graphs_4 = copy.deepcopy(train_request_HamS_model)
train_request_HamS_different_artificial_graphs_4.arguments["datamodule_hyperparams"]["train_expected_noise_edges_per_node"] = 4

train_request_HamS_different_artificial_graphs_2 = copy.deepcopy(train_request_HamS_model)
train_request_HamS_different_artificial_graphs_2.arguments["datamodule_hyperparams"]["train_expected_noise_edges_per_node"] = 2