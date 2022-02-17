from asyncio.log import logger
import pytorch_lightning as torch_lightning
import pytorch_lightning.loggers as lightning_loggers
import datetime

import src.data.DataModules as DataModules
import src.constants as constants
import src.Models as Models

def train_model(model_class, datamodule_class, model_checkpoint=None, model_hyperparams={}, datamodule_hyperparams={},
                trainer_hyperparams={}, train_parameters={},
                checkpoint_saving_dir=constants.MODEL_CHECKPOINT_SAVING_DIRECTORY, wandb_directory=constants.WANDB_DIRECTORY,
                is_log_offline=True, wandb_project=constants.WEIGHTS_AND_BIASES_PROJECT, run_name=None, is_log_model=False):
    if run_name is None:
        datetime_now = datetime.datetime.now()
        run_name = f"{model_class}_{datetime_now.strftime('%Y%m%d%h')[2:]}"
    wandb_logger = lightning_loggers.WandbLogger(name=run_name, project=wandb_project, offline=is_log_offline, log_model=is_log_model)
    # wandb_logger.__init__()
    wandb_logger.experiment.log({"is_started_from_checkpoint": model_checkpoint is not None,
                                 "checkpoint": model_checkpoint})

    if model_checkpoint is None:
        model = model_class(**model_hyperparams)
    else:
        model = model_class.load_from_checkpoint(model_checkpoint)
    wandb_logger.experiment.log({"description": model.description()})

    datamodule_hyperparams.update({
        DataModules.SIMULATION_MODULE_VARIABLE_NAME: model
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

    wandb_logger.finalize()

def test_model():
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

if __name__== "__main__":
    model_class = Models.GatedGCNEmbedAndProcess
    model_hyperparams = {"processor_depth": 5}

    datamodule_class = DataModules.ReinforcementErdosRenyiDataModule
    datamodule_hyperparams = {}

    trainer_hyperparams = {"max_epochs": 10, "num_sanity_val_steps": 2}

    train_model(model_class, datamodule_class, model_hyperparams=model_hyperparams, trainer_hyperparams=trainer_hyperparams)
