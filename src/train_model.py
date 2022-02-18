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
                is_log_offline=False, wandb_project=constants.WEIGHTS_AND_BIASES_PROJECT, run_name=None, is_log_model=True):
    if run_name is None:
        datetime_now = datetime.datetime.now()
        run_name = f"{model_class.__name__.split('.')[-1][:10]}_{datetime_now.strftime('%Y%m%d%H%M')[2:]}"
    wandb_logger = lightning_loggers.WandbLogger(name=run_name, project=wandb_project, offline=is_log_offline, log_model=is_log_model)
    wandb_logger.experiment.summary["is_started_from_checkpoint"] = model_checkpoint is not None
    wandb_logger.experiment.summary["checkpoint"] = str(model_checkpoint)

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

    wandb_logger.finalize()

if __name__== "__main__":
    model_class = Models.GatedGCNEmbedAndProcess
    model_hyperparams = {"processor_depth": 3}

    datamodule_class = DataModules.ReinforcementErdosRenyiDataModule
    datamodule_hyperparams = {}

    trainer_hyperparams = {"max_epochs": 10, "num_sanity_val_steps": 2, "log_every_n_steps": 20}
    datamodule_hyperparams = {"train_virtual_epoch_size": 1000}

    train_model(model_class, datamodule_class, model_hyperparams=model_hyperparams, trainer_hyperparams=trainer_hyperparams,
                datamodule_hyperparams=datamodule_hyperparams)
