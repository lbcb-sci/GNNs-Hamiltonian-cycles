import datetime
import itertools
from inspect import isclass
import tempfile
from pathlib import Path

import torch
import pytorch_lightning as torch_lightning
import pytorch_lightning.loggers as lightning_loggers
import pytorch_lightning.callbacks as lightning_callbacks
import wandb

from hamgnn.Evaluation import EvaluationScores
from hamgnn.HamiltonSolver import HamiltonSolver
import hamgnn.data.DataModules as DataModules
import hamgnn.constants as constants
import hamgnn.nn_modules.hamilton_gnn_utils as gnn_utils


def train_model(model_class, datamodule_class, model_checkpoint=None, model_hyperparams=None, datamodule_hyperparams=None,
                trainer_hyperparams=None, train_parameters=None, model_checkpoint_hyperparams=None,
                is_log_offline=False, wandb_project=constants.WEIGHTS_AND_BIASES_PROJECT, run_name=None, is_log_model=True,
                is_run_test_at_the_end=False):
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

    if model_checkpoint is None:
        model = model_class(**model_hyperparams)
    else:
        model = model_class.load_from_checkpoint(model_checkpoint)

    datamodule_hyperparams.update({
        DataModules.LIGHTNING_MODULE_REFERENCE_KEYWORD: model,
    })
    datamodule = datamodule_class(**datamodule_hyperparams)

    trainer_hyperparams.update({
        "default_root_dir": Path(constants.MODEL_CHECKPOINT_SAVING_DIRECTORY).resolve()
    })

    wandb_logger = lightning_loggers.WandbLogger(name=run_name, project=wandb_project, offline=is_log_offline, log_model=is_log_model)
    wandb_logger.experiment.config["is_started_from_checkpoint"] = model_checkpoint is not None
    for key, value in itertools.chain(datamodule_hyperparams.items(), trainer_hyperparams.items()):
        wandb_logger.experiment.config[key] = value
    wandb_logger.experiment.summary["description"] = model.description()
    trainer_hyperparams["logger"] = wandb_logger

    model_checkpoint_hyperparams["save_last"] = True
    failsafe_checkpoint_callback = lightning_callbacks.ModelCheckpoint(**model_checkpoint_hyperparams)
    callbacks = [failsafe_checkpoint_callback]
    if "callbacks" in trainer_hyperparams:
            trainer_hyperparams["callbacks"].extend(callbacks)
    else:
        trainer_hyperparams["callbacks"] = callbacks
    checkpoint_callback = [cb for cb in  trainer_hyperparams["callbacks"] if isinstance(cb, lightning_callbacks.ModelCheckpoint)][0]

    trainer = torch_lightning.Trainer(**trainer_hyperparams)

    if "auto_lr_find" in trainer_hyperparams:
        print("Automatic lr enabled, computing...")
        lr_find_kwargs = {"min_lr": 1e-9, "max_lr": 1e-2, "num_training": 100, "early_stop_threshold": 5, "mode":"linear"}
        lr_finder = trainer.tune(model, datamodule=datamodule, lr_find_kwargs=lr_find_kwargs)["lr_find"]
        fig = lr_finder.plot(suggest=True)
        suggested_lr = lr_finder.suggestion()
        print(f"Using suggested lr: {suggested_lr}")
        # TODO remove wandb.Image after installing plotly into env. Wandb crashes without plotly
        wandb_logger.experiment.log({"lr_find_plot": wandb.Image(fig)})
        wandb_logger.experiment.config.update({"adjusted_learning_rate": model.learning_rate})

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
        if is_run_test_at_the_end:
            test_on_saved_data(model, wandb_logger.experiment)
        wandb_logger.finalize("Success")
        if train_exception is not None:
            raise train_exception
    return model, checkpoint_callback.best_model_path

def reconnect_to_wandb_run(wandb_run_id):
    return wandb.init(id=wandb_run_id, project=constants.WEIGHTS_AND_BIASES_PROJECT, resume=True)


def create_model_from_checkpoint(checkpoint_path):
    model_classes = gnn_utils.list_of_gnn_model_classes()
    model = None
    for c in model_classes:
        try:
            model = c.load_from_checkpoint(checkpoint_path)
            break
        except:
            pass
    return model


def create_model_for_wandb_run(wandb_run, checkpoint_path=None):
    model = None
    if "checkpoint" in wandb_run.config:
        try:
            checkpoint_path = wandb_run.config["checkpoint"]
            model = create_model_from_checkpoint(checkpoint_path)
        except Exception as ex:
            model = None

    if model is None:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                artifact = wandb_run.use_artifact(f"model-{wandb_run.id}:v0")
                artifact.download(tmp_dir)
                # Hacky solution. W&B documentation is not very clear on how these artifcats should be used
                checkpoint_path = Path(tmp_dir) / "model.ckpt"
                model = create_model_from_checkpoint(checkpoint_path)
        except Exception as ex:
            model = None

    return model

def load_existing_model(model_identifier, wandb_project=constants.WEIGHTS_AND_BIASES_PROJECT, wandb_run=None):
    checkpoint_path = Path(model_identifier)
    model = None
    if checkpoint_path.exists():
        try:
            model = create_model_from_checkpoint(checkpoint_path)
            if model is None:
                return None, f"Failed to create model from {checkpoint_path}. Appropriate model classes seem to be missing."
            else:
                return model, f"OK"
        except Exception as ex:
            model = None

    wandb_run = None
    if model is None:
        try:
            wandb_id = model_identifier
            wandb_run = wandb.init(project=wandb_project, id=wandb_id, resume=True)
            model = create_model_for_wandb_run(wandb_run, wandb_run.config["checkpoint"])
            if model is None:
                return None, "Identified wandb run but failed to create the model for it"
            else:
                return model, "OK"
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    return None, "Failed"


def test_on_saved_data(model: HamiltonSolver, wandb_run=None, store_tag=constants.DEFAULT_FINAL_TEST_TAG):
    df_testing_results = EvaluationScores.accuracy_scores_on_saved_data([model], ["model"], nr_graphs_per_size=None, is_show_progress=True)

    if wandb_run is not None:
        for row_index, row in df_testing_results.iterrows():
            for accuracy_tag in [EvaluationScores.ACCURACY_SCORE_TAGS.perc_hamilton_found, EvaluationScores.ACCURACY_SCORE_TAGS.perc_long_cycles_found,
                                EvaluationScores.ACCURACY_SCORE_TAGS.perc_full_walks_found, EvaluationScores.ACCURACY_SCORE_TAGS.perc_long_walks_found,
                                EvaluationScores.ACCURACY_SCORE_TAGS.avg_execution_time]:
                wandb_run.log({f"{store_tag}/graph_size": row["graph size"], f"{store_tag}/{accuracy_tag}": row[accuracy_tag]})
    return df_testing_results


class ModelTrainRequest:
    def __init__(self, **kwargs):
        self.arguments = kwargs

    def train(self, nr_cpu_threads=16, run_name=None, is_run_test_at_the_end=False):
        torch.set_num_threads(nr_cpu_threads)
        extended_arguments = self.arguments.copy()
        if "run_name" not in self.arguments or self.arguments["run_name"] is None:
            extended_arguments.update({
                "run_name": run_name,
                "is_run_test_at_the_end": is_run_test_at_the_end})
        return train_model(**extended_arguments)
