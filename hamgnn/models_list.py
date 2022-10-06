import copy
from pathlib import Path

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from hamgnn.nn_modules.EncodeProcessDecodeNN import EncodeProcessDecodeAlgorithm
from hamgnn.nn_modules.EncodeProcessDecodeNoHiddenNN import _EncodeProcessDecodeNoHidden
from hamgnn.nn_modules.EmbeddingAndMaxMPNN import EmbeddingAndMaxMPNN
import hamgnn.nn_modules.EncodeProcessDecodeWithLayerNorm as EncodeProcessDecodeWithLayerNorm
from hamgnn.nn_modules.EncodeProcessDecodeRandFeatures import EncodeProcessDecodeRandFeatures
import hamgnn.data.DataModules as DataModules
import hamgnn.model_utils as model_utils
import hamgnn.callbacks as my_callbacks

train_request_HamS = model_utils.ModelTrainRequest(
    model_class = EncodeProcessDecodeAlgorithm,
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
train_request_HamS.arguments["trainer_hyperparams"]["callbacks"] = lr_callbacks + norm_monitoring_callbacks + checkpoint_callbacks


train_request_HamS_quick = copy.deepcopy(train_request_HamS)
train_request_HamS_quick.arguments["trainer_hyperparams"].update({"max_epochs": 50})

train_request_HamS_mse_loss = copy.deepcopy(train_request_HamS)
train_request_HamS_mse_loss.arguments["model_hyperparams"].update({"loss_type": "mse"})


train_request_HamS_depth_3 = copy.deepcopy(train_request_HamS)
train_request_HamS_depth_3.arguments["model_hyperparams"].update({"processor_depth": 3})

train_request_HamS_depth_7 = copy.deepcopy(train_request_HamS)
train_request_HamS_depth_7.arguments["model_hyperparams"].update({"processor_depth": 7})

train_request_HamS_depth_15 = copy.deepcopy(train_request_HamS)
train_request_HamS_depth_15.arguments["model_hyperparams"].update({"processor_depth": 15})


train_request_HamS_graphs_50 = copy.deepcopy(train_request_HamS)
train_request_HamS_graphs_50.arguments["datamodule_hyperparams"].update({"train_graph_size": 50, "val_graph_size": 50})

train_request_HamS_graphs_100 = copy.deepcopy(train_request_HamS)
train_request_HamS_graphs_100.arguments["datamodule_hyperparams"].update({"train_graph_size": 100, "val_graph_size": 100})

train_request_HamS_graphs_200 = copy.deepcopy(train_request_HamS)
train_request_HamS_graphs_200.arguments["datamodule_hyperparams"].update({"train_graph_size": 200, "val_graph_size": 200})


train_request_HamS_ER_exact_solver = copy.deepcopy(train_request_HamS)
train_request_HamS_ER_exact_solver.arguments["datamodule_class"] = DataModules.SolvedErdosRenyiDataModule
train_request_HamS_ER_exact_solver.arguments["datamodule_hyperparams"].update({"train_hamilton_existence_probability": 0.8})
del train_request_HamS_ER_exact_solver.arguments["datamodule_hyperparams"]["train_expected_noise_edges_per_node"]

train_request_HamS_grad_clipping = copy.deepcopy(train_request_HamS)
train_request_HamS_grad_clipping.arguments["trainer_hyperparams"]["gradient_clip_val"] = 1

train_request_HamS_custom_lr = copy.deepcopy(train_request_HamS)
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

train_request_HamS_different_artificial_graphs_4 = copy.deepcopy(train_request_HamS)
train_request_HamS_different_artificial_graphs_4.arguments["datamodule_hyperparams"]["train_expected_noise_edges_per_node"] = 4

train_request_HamS_different_artificial_graphs_2 = copy.deepcopy(train_request_HamS)
train_request_HamS_different_artificial_graphs_2.arguments["datamodule_hyperparams"]["train_expected_noise_edges_per_node"] = 2

train_request_HamS_different_artificial_graphs_3_5 = copy.deepcopy(train_request_HamS)
train_request_HamS_different_artificial_graphs_3_5.arguments["datamodule_hyperparams"]["train_expected_noise_edges_per_node"] = 3.5

train_request_HamS_mse = copy.deepcopy(train_request_HamS)
train_request_HamS_mse.arguments["model_hyperparams"]["loss_type"] = "mse"

train_request_HamS_mse_large = copy.deepcopy(train_request_HamS_mse)
train_request_HamS_mse_large.arguments["datamodule_hyperparams"].update({"train_graph_size": 200, "val_graph_size": 200})

train_request_HamS_mse_large_faster = copy.deepcopy(train_request_HamS_mse_large)
train_request_HamS_mse_large_faster.arguments["model_hyperparams"]["starting_learning_rate"] *= 20


train_request_HamS_no_hidden = copy.deepcopy(train_request_HamS)
train_request_HamS_no_hidden.arguments["model_class"] = _EncodeProcessDecodeNoHidden

train_request_HamS200_no_hidden = copy.deepcopy(train_request_HamS_no_hidden)
train_request_HamS200_no_hidden.arguments["datamodule_hyperparams"]["train_graph_size"] = 200


train_request_HamS_rare_artificial_cycle = copy.deepcopy(train_request_HamS)
train_request_HamS_rare_artificial_cycle.arguments["datamodule_class"] = DataModules.ArtificialCycleDataModule
train_request_HamS_rare_artificial_cycle.arguments["datamodule_hyperparams"] = {
    "train_virtual_epoch_size": 8 * 100,
    "val_virtual_epoch_size": 8 * 100,
    "train_batch_size": 8,
    "val_batch_size": 8,
    "train_graph_size": 50,
    "val_graph_size": 50,
    "train_expected_noise_edges_per_node": 1.3,
    "val_expected_noise_edges_per_node": 1.3
}
train_request_HamS_rare_artificial_cycle.arguments["trainer_hyperparams"]["max_epochs"] = 200
_rare_artificial_checkpoint_callbacks = [ModelCheckpoint(save_top_k=3, save_last=True, monitor="val/artificial/hamiltonian_cycle")]
train_request_HamS_rare_artificial_cycle.arguments["trainer_hyperparams"]["callbacks"] = lr_callbacks + norm_monitoring_callbacks + _rare_artificial_checkpoint_callbacks


train_request_HamS_rare_small = copy.deepcopy(train_request_HamS_rare_artificial_cycle)
train_request_HamS_rare_small.arguments["datamodule_hyperparams"].update({
    "train_graph_size": 25,
    "val_graph_size": 25,
})
train_request_HamS_rare_small.arguments["trainer_hyperparams"].update({"max_epochs": 500})

train_request_HamS_rare_really_small = copy.deepcopy(train_request_HamS_rare_small)
train_request_HamS_rare_really_small.arguments["datamodule_hyperparams"].update({
    "train_expected_noise_edges_per_node": 0.07,
    "val_expected_noise_edges_per_node": 0.07
})

# GPU training
train_request_HamS_gpu = copy.deepcopy(train_request_HamS)
train_request_HamS_gpu.arguments["datamodule_class"] = DataModules.ArtificialCycleDataModule
del train_request_HamS_gpu.arguments["datamodule_hyperparams"]["val_hamiltonian_existence_probability"]
train_request_HamS_gpu.arguments["model_hyperparams"]["val_dataloader_tags"] = ["artificial"]
train_request_HamS_gpu.arguments["trainer_hyperparams"].update({
    "gpus": [0]
})
train_request_HamS_gpu.arguments["datamodule_hyperparams"].update({"train_batch_size": 16, "val_batch_size": 8})
train_request_HamS_gpu.arguments["trainer_hyperparams"]["callbacks"] = train_request_HamS_gpu.arguments["trainer_hyperparams"]["callbacks"][:-1] \
    + [ModelCheckpoint(save_top_k=3, save_last=True, monitor="val/artificial/hamiltonian_cycle")]

train_request_HamS_gpu_layer_norm = copy.deepcopy(train_request_HamS_gpu)
train_request_HamS_gpu_layer_norm.arguments["model_class"] = EncodeProcessDecodeWithLayerNorm.EncodeProcessDecodeWithLayerNorm

train_request_HamS_gpu_large = copy.deepcopy(train_request_HamS_gpu_layer_norm)
train_request_HamS_gpu.arguments["trainer_hyperparams"].update({"max_epochs": 2000})
train_request_HamS_gpu_large.arguments["model_hyperparams"].update(
    {"processor_depth": 7,
    "hidden_dim": 128}
)

train_request_HamS_gpu_large_size_50 = copy.deepcopy(train_request_HamS_gpu_large)
train_request_HamS_gpu_large_size_50.arguments["datamodule_hyperparams"].update({
    "train_graph_size": 50
})

# GNN operations include random features
train_request_HamS_gpu_with_rand_node_encoding = copy.deepcopy(train_request_HamS_gpu)
train_request_HamS_gpu_with_rand_node_encoding.arguments["model_class"] = EncodeProcessDecodeRandFeatures

# Improved environment input embedding
from hamgnn.nn_modules.EncodeProcessDecodeAdvancedPositionalEmbedding import EncodeProcessDecodeAdvancedPositionalEmbedding
train_request_HamS_gpu_advanced_input = copy.deepcopy(train_request_HamS_gpu)
train_request_HamS_gpu_advanced_input.arguments["model_class"] = EncodeProcessDecodeAdvancedPositionalEmbedding

# Testing batch size
train_request_HamS_gpu_large_batch = copy.deepcopy(train_request_HamS_gpu)
train_request_HamS_gpu_large_batch.arguments["datamodule_hyperparams"].update({
    "train_batch_size": 64
})
train_request_HamS_gpu.arguments["trainer_hyperparams"].update({
    "gpus": [0]
})



# # Training on genomic data
# import hamgnn.data.genomic_datasets as genomic_datasets
# data_root_folder = (Path(__file__).parent / "../genome_graphs/SnakemakePipeline").resolve()
# train_request_HamS_genomes = copy.deepcopy(train_request_HamS)
# train_request_HamS_genomes.arguments["model_hyperparams"].update({
#     "val_dataloader_tags": None,
#     "loss_type": "mse"
# })
# train_request_HamS_genomes.arguments["datamodule_class"] = genomic_datasets.StringGraphDatamodule
# train_request_HamS_genomes.arguments["datamodule_hyperparams"] = {
#     "train_batch_size": 1,
#     "val_batch_size": 1,
#     "train_paths": (data_root_folder / "string_graphs_train.txt").read_text().split(),
#     "val_paths": (data_root_folder / "string_graphs_val.txt").read_text().split(),
#     "test_paths": (data_root_folder / "string_graphs_test.txt").read_text().split()
# }
# train_request_HamS_genomes.arguments["trainer_hyperparams"]["callbacks"] = lr_callbacks + norm_monitoring_callbacks + [ModelCheckpoint(save_top_k=3, save_last=True, monitor="val/hamiltonian_cycle")]

# train_request_HamS_genomes_lr = copy.deepcopy(train_request_HamS_genomes)
# train_request_HamS_genomes_lr.arguments["model_hyperparams"]["starting_learning_rate"] = 1e-7



### Reinforcement learning models

train_request_HamR = model_utils.ModelTrainRequest(
    # is_log_offline=True,
    # is_log_model=False,
    model_class = EmbeddingAndMaxMPNN,
    datamodule_class = DataModules.ReinforcementErdosRenyiDataModule,
    model_checkpoint = None,
    model_hyperparams = {
        "processor_depth": 5,
        "loss_type": "entropy",
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
        "train_ham_existence_prob": 0.8,
    },
    model_checkpoint_hyperparams = {
        "every_n_epochs": 1,
        "dirpath": None,
    }
)