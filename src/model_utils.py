import datetime
from gc import callbacks
import itertools

import torch
import pytorch_lightning as torch_lightning
import pytorch_lightning.loggers as lightning_loggers
import pytorch_lightning.callbacks as lightning_callbacks

import src.Models as models
from src.Evaluation import EvaluationScores
from src.HamiltonSolver import HamiltonSolver
import src.data.DataModules as DataModules
import src.constants as constants


def train_model(model_class, datamodule_class, model_checkpoint=None, model_hyperparams=None, datamodule_hyperparams=None,
                trainer_hyperparams=None, train_parameters=None, model_checkpoint_hyperparams=None,
                is_log_offline=False, wandb_project=constants.WEIGHTS_AND_BIASES_PROJECT, run_name=None, is_log_model=True):
    if model_hyperparams is None:
        model_hyperparams = {}
    if datamodule_hyperparams is None:
        datamodule_hyperparams = {}
    if trainer_hyperparams is None:
        trainer_hyperparams = {}
    if train_parameters is None:
        train_parameters = {}
    if model_checkpoint_hyperparams is None:
        model_checkpoint_hyperparams = {}

    if run_name is None:
        datetime_now = datetime.datetime.now()
        run_name = f"{model_class.__name__.split('.')[-1][:10]}_{datetime_now.strftime('%Y%m%d%H%M')[2:]}"


    wandb_logger = lightning_loggers.WandbLogger(name=run_name, project=wandb_project, offline=is_log_offline, log_model=is_log_model)
    wandb_logger.experiment.config["is_started_from_checkpoint"] = model_checkpoint is not None
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

    model_checkpoint_hyperparams["save_last"] = True
    checkpoint_callback = lightning_callbacks.ModelCheckpoint(**model_checkpoint_hyperparams)
    callbacks = [checkpoint_callback]
    trainer_hyperparams["callbacks"] = callbacks

    trainer = torch_lightning.Trainer(**trainer_hyperparams)

    train_parameters.update({
        "model": model,
        "datamodule": datamodule
    })
    train_exception = None
    try:
        trainer.fit(**train_parameters)
    except Exception as ex:
        train_exception = ex
    finally:
        wandb_logger.experiment.config["checkpoint"] = str(checkpoint_callback.best_model_path)
        test_on_saved_data(model, wandb_logger.experiment)
        wandb_logger.finalize("Success")
        if train_exception is not None:
            raise train_exception


def create_model_for_wandb_run(wandb_run, checkpoint_path):
    checkpoint = wandb_run.config["checkpoint"]
    model_classes = [var for var_name, var in vars(models) if isinstance(var, models.HamFinderGNN)]
    model = None
    for c in model_classes:
        try:
            model = c.load_from_checkpoint(checkpoint)
        except:
            pass
    return model


def test_on_saved_data(model: HamiltonSolver, wandb_run=None):
    df_testing_results = EvaluationScores.accuracy_scores_on_saved_data([model], ["model"])

    unified_test_tag = "unified_test"

    if wandb_run is not None:
        for row_index, row in df_testing_results.iterrows():
            for accuracy_tag in [EvaluationScores.ACCURACY_SCORE_TAGS.perc_hamilton_found, EvaluationScores.ACCURACY_SCORE_TAGS.perc_long_cycles_found,
                                EvaluationScores.ACCURACY_SCORE_TAGS.perc_full_walks_found, EvaluationScores.ACCURACY_SCORE_TAGS.perc_long_walks_found]:
                wandb_run.log({f"{unified_test_tag}/graph_size": row["graph size"], f"{unified_test_tag}/{accuracy_tag}": row[accuracy_tag]})
    return df_testing_results


class ModelTrainRequest:
    def __init__(self, **kwargs):
        self.arguments = kwargs

    def train(self, nr_cpu_threads=16):
        torch.set_num_threads(nr_cpu_threads)
        train_model(**self.arguments)
