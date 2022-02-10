import os

from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as torch_lightning
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning.callbacks
import torch_geometric
import wandb

from src.Evaluation import EvaluationScores
import src.Models as Models
from src.data.InMemoryDataset import ErdosRenyiInMemoryDataset
from src.data.GraphGenerators import ErdosRenyiGenerator, NoisyCycleGenerator
from src.data.GraphDataset import GraphDataLoader, GraphGeneratingDataset
from src.data.DataModules import ArtificialCycleDataModule, ErdosRenyiInMemoryDataset, ReinforcementErdosRenyiDataModule


def check_existing_models_training(wandb_project=None, weights_dir_path=Path("new_weights")):
    torch.set_num_threads(8)

    if wandb_project is None:
        wandb_project = "Unnamed_project"
        is_wandb_log_offline = True
    else:
        is_wandb_log_offline = False

    HamR = Models.GatedGCNEmbedAndProcess.load_from_checkpoint(weights_dir_path / "lightning_HamR.ckpt")
    HamR_datamodule = ReinforcementErdosRenyiDataModule(HamR)
    HamR_trainer = torch_lightning.Trainer(max_epochs=0, num_sanity_val_steps=2, logger=WandbLogger(project=wandb_project, offline=is_wandb_log_offline))
    HamR_trainer.fit(HamR, datamodule=HamR_datamodule)
    test_dataloaders = {int(path.split('(')[1].split(',')[0]): GraphDataLoader(ErdosRenyiInMemoryDataset([os.path.join("DATA", path)]), 1) for path in os.listdir("DATA") if path.endswith(".pt")}
    test_dataloaders_list = [GraphDataLoader(ErdosRenyiInMemoryDataset([os.path.join("DATA", path)]), 1) for path in os.listdir("DATA") if path.endswith(".pt")]
    _original_log_test_tag = HamR.log_test_tag
    for graph_size, dataloader in test_dataloaders.items():
        if graph_size > 100:
            continue
        HamR.log_test_tag = f"test_graph_size_{graph_size}"
        HamR_trainer.test(HamR, dataloaders=[dataloader])
    HamR.log_test_tag =  _original_log_test_tag
    wandb.finish()

    HamS = Models.EncodeProcessDecodeAlgorithm.load_from_checkpoint(weights_dir_path / "lightning_HamS.ckpt")
    HamS_datamodule = ArtificialCycleDataModule()
    HamS_trainer = torch_lightning.Trainer(max_epochs=0, num_sanity_val_steps=2, logger=WandbLogger(project=wandb_project, offline=is_wandb_log_offline))
    HamS_trainer.fit(HamS, datamodule=HamS_datamodule)
    _original_log_test_tag = HamS.log_test_tag
    for graph_size, dataloader in test_dataloaders.items():
        if graph_size > 100:
            continue
        HamS.log_test_tag = f"test_graph_size_{graph_size}"
        HamS_trainer.test(HamS, dataloaders=[dataloader])
    HamS.log_test_tag = _original_log_test_tag
    wandb.finish()
    exit()

    # print("Testing...")
    # import itertools
    # dataset = ErdosRenyiInMemoryDataset(["tst/small_data_sample"])
    # graphs = [example.graph for example in itertools.islice(iter(dataset), 10)]

    # def solve_fn(actual_fn, graphs):
    #     walks = []
    #     for g in iter(graphs):
    #         solutions_tensor = actual_fn(torch_geometric.data.Batch.from_data_list([g]))
    #         walks.append([x.item() for x in solutions_tensor[0, :] if x.item() != -1])
    #     return walks

    # old_solutions = solve_fn(lambda _: HamS.batch_run_greedy_neighbor_old(_)[0], graphs)
    # new_solutions = solve_fn(HamS.batch_run_greedy_neighbor, graphs)

    # old_evals = EvaluationScores.evaluate_on_saved_data(
    #     lambda graphs: solve_fn(lambda _: HamS.batch_run_greedy_neighbor_old(_)[0], graphs), nr_graphs_per_size=20, data_folders=["tst/small_data_sample"])
    # new_evals = EvaluationScores.evaluate_on_saved_data(
    #     lambda graphs: solve_fn(HamS.batch_run_greedy_neighbor, graphs), nr_graphs_per_size=20, data_folders=["tst/small_data_sample"])
    # old_accuracy = EvaluationScores.compute_accuracy_scores(old_evals)
    # print(old_accuracy)
    # new_accuracy = EvaluationScores.compute_accuracy_scores(new_evals)
    # print(new_accuracy)

    df = EvaluationScores.accuracy_scores_on_saved_data([HamS], "HamS", nr_graphs_per_size=100)
    print(df)
    print("Runing lightning test...")
    tmp = HamS_trainer.test(HamS, datamodule=HamS_datamodule)
    print(tmp)


    HamR = Models.GatedGCNEmbedAndProcess.load_from_checkpoint(weights_dir_path / "lightning_HamR.ckpt")
    HamR_datamodule = ReinforcementErdosRenyiDataModule(HamR)
    HamR_trainer = torch_lightning.Trainer(max_epochs=10, num_sanity_val_steps=2)
    test_results = HamR_trainer.fit(HamR, datamodule=HamR_datamodule)
    print(test_results)


if __name__ == "__main__":
    torch.set_num_threads(32)
    wandb_project = "gnns-Hamiltonian-cycles"

    check_existing_models_training(wandb_project=wandb_project)

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
