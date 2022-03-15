import datetime
import itertools
import time
from typing import Mapping
from numpy import size

import torch
import pytorch_lightning as torch_lightning
import pytorch_lightning.loggers as lightning_loggers
import wandb

from src.Evaluation import EvaluationScores
from src.HamiltonSolver import HamiltonSolver
from src.data.GraphDataset import GraphDataLoader, GraphExample, GraphGeneratingDataset
import src.data.DataModules as DataModules
import src.constants as constants
from src.data.InMemoryDataset import ErdosRenyiInMemoryDataset


def train_model(model_class, datamodule_class, model_checkpoint=None, model_hyperparams={}, datamodule_hyperparams={},
                trainer_hyperparams={}, train_parameters={},
                checkpoint_saving_dir=constants.MODEL_CHECKPOINT_SAVING_DIRECTORY, wandb_directory=constants.WANDB_DIRECTORY,
                is_log_offline=False, wandb_project=constants.WEIGHTS_AND_BIASES_PROJECT, run_name=None, is_log_model=True):
    if run_name is None:
        datetime_now = datetime.datetime.now()
        run_name = f"{model_class.__name__.split('.')[-1][:10]}_{datetime_now.strftime('%Y%m%d%H%M')[2:]}"


    wandb_logger = lightning_loggers.WandbLogger(name=run_name, project=wandb_project, offline=is_log_offline, log_model=is_log_model)
    wandb_logger.experiment.config["is_started_from_checkpoint"] = model_checkpoint is not None
    wandb_logger.experiment.config["checkpoint"] = str(model_checkpoint)
    for key, value in itertools.chain(datamodule_hyperparams.items(), trainer_hyperparams.items()):
        wandb_logger.experiment.config[key] = value

    if model_checkpoint is None:
        model = model_class(**model_hyperparams)
    else:
        model = model_class.load_from_checkpoint(model_checkpoint)
    wandb_logger.experiment.summary["description"] = model.description()

    datamodule_hyperparams.update({
        DataModules.LIGHTNING_MODULE_REFERENCE: model,
    })
    datamodule = datamodule_class(**datamodule_hyperparams)

    trainer_hyperparams.update({
        "logger": wandb_logger
    })
    trainer = torch_lightning.Trainer(**trainer_hyperparams)

    train_parameters.update({
        "model": model,
        "datamodule": datamodule
    })
    trainer.fit(**train_parameters)

    trainer.test(datamodule=datamodule)
    wandb_logger.finalize("Success")


def test_on_saved_data(model_class: HamiltonSolver, wandb_run=None, model_hyperparams={}, model_checkpoint=None):
    if model_checkpoint is None:
        model = model_class(model_hyperparams)
    else:
        model = model_class.load_from_checkpoint()
    df_testing_results = EvaluationScores.accuracy_scores_on_saved_data([model], ["model"])

    unified_test_tag = "unified_test"

    if wandb_run is not None:
        for row_index, row in df_testing_results.iterrows():
            for accuracy_tag in [EvaluationScores.ACCURACY_SCORE_TAGS.perc_hamilton_found, EvaluationScores.ACCURACY_SCORE_TAGS.perc_long_cycles_found,
                                EvaluationScores.ACCURACY_SCORE_TAGS.perc_full_walks_found, EvaluationScores.ACCURACY_SCORE_TAGS.perc_long_walks_found]:
                wandb_run.log({f"{unified_test_tag}/size": row["size"], f"{unified_test_tag}/{accuracy_tag}": row[accuracy_tag]})
    return df_testing_results


def test_model_on_saved_data(model_class, model_hyperparams = {}, model_checkpoint=None, wandb_run_id=None):
    dataset = ErdosRenyiInMemoryDataset([constants.EVALUATION_DATA_FOLDERS])

    sizes = set(graph_example.graph.num_nodes for graph_example in dataset)
    name_to_datasets = {
        size: (graph_example for graph_example in dataset if graph_example.graph.num_nodes == size)
        for size in sizes
    }

    return 0
    timestamp = time.time()
    # TODO add wandb logging connection

    df = EvaluationScores.accuracy_scores_on_saved_data(
        [HamS_model, HamR_model, hybrid_ham_heuristics, least_degree_first, concorde_solver],
        ["HamS", "HamR", "HybridHam", "Least_degree_first", "Concorde"], nr_graphs_per_size=10_000)
    df["timestamp"] = timestamp
    dataset = ErdosRenyiInMemoryDataset(constants.EVALUATION_DATA_FOLDERS)
    sizes = list(set(graph_item.graph.num_nodes for graph_item in dataset))
    test_dataloaders = []
    for size in sizes:
        subset_dataset = copy.deepcopy(dataset)
        subset_dataset.data_list = \
            [data_item for data_item in dataset.data_list if data_item[ErdosRenyiInMemoryDataset.NUM_NODES_TAG] == size]
        test_dataloaders.append(GraphDataset.GraphDataLoader(subset_dataset))

    results = {}
    trainer = pytorch_lightning.Trainer()
    for size, test_dataloader in zip(size, test_dataloaders):
         results[size] = trainer.test(model, dataloaders=test_dataloaders[1])
    return results


class ModelTrainRequest:
    def __init__(self, **kwargs):
        self.arguments = kwargs

    def train(self, nr_cpu_threads=16):
        torch.set_num_threads(nr_cpu_threads)
        train_model(**self.arguments)
