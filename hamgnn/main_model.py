from pathlib import Path
import shutil

import torch

import hamgnn.constants as constants
import hamgnn.model_utils as model_utils
import hamgnn.models_list as models_list

MAIN_MODEL_CHECKPOINT_PATH = constants.MAIN_MODEL_CHECKPOINT_PATH
MAIN_MODEL_CHECKPOINT_NAME = constants.MAIN_MODEL_CHECKPOINT_PATH.name
MAIN_MODEL_TRAIN_REQUEST = models_list.train_request_HamS_gpu_with_rand_node_encoding

ABLATION_NO_RANDOM_FEATURES_CHECKPOINT_PATH = constants.ABLATION_NO_RANDOM_FEATURES_PATH
ABLATION_NO_RANDOM_FEATURES_TRAIN_REQUEST = models_list.train_request_HamS_gpu
ABLATION_NO_HIDDEN_FEATURES_CHECKPOINT_PATH = constants.ABLATION_NO_PERSISTENT_FEATURES_PATH
ABLATION_NO_HIDDEN_FEATURES_TRAIN_REQUEST = models_list.train_request_HamS_gpu_no_hidden_with_random_node_encoding
ABLATION_MAIN_MODEL_CHECKPOINT_PATH = constants.ABLATION_MAIN_MODEL_PATH


def store_as_main_model(checkpoint_path):
    shutil.copyfile(checkpoint_path, MAIN_MODEL_CHECKPOINT_PATH)


def train_main_model(variation_nr=None):
    run_name = "main_model"
    if variation_nr is not None:
        run_name = f"{run_name}_{variation_nr}"
    return MAIN_MODEL_TRAIN_REQUEST.train(nr_cpu_threads=1, run_name=run_name)


def _yes_no_input(message):
    AFFIRMATIVE = ["yes", "y"]
    NEGATIVE = ["no", "n"]
    while True:
        print(message)
        response = input()
        response = response.strip().lower()
        if response in AFFIRMATIVE:
            return True
        elif response in NEGATIVE:
            return False
        else:
            print("Failed to understand the input. Please respond with 'yes'/'y' or 'no'/'n'.")


def _load_model(checkpoint_path, train_model_fn, name="model"):
    if checkpoint_path.exists():
        model, load_message = model_utils.load_existing_model(checkpoint_path)
    else:
        is_train = _yes_no_input(f"The {name} does not exist yet. Do you wish to train it?")
        if is_train:
            model, tmp_checkpoint_path = train_model_fn()
            shutil.copyfile(tmp_checkpoint_path, checkpoint_path)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def load_main_model():
    model_name = "main model"
    checkpoint_path = MAIN_MODEL_CHECKPOINT_PATH
    return _load_model(checkpoint_path, train_main_model, model_name)


def load_ablation_main_model(variation_nr=None):
    model_name = "ablation main model"
    checkpoint_path = ABLATION_MAIN_MODEL_CHECKPOINT_PATH
    if variation_nr is not None:
        model_name = f"{model_name}_{variation_nr}"
        checkpoint_path = get_model_variation_checkpoint_path(checkpoint_path, variation_nr)
    return _load_model(checkpoint_path, lambda: train_main_model(variation_nr), model_name)


def train_ablation_no_random_features(variation_nr=None):
    run_name = "ablation_no_random_features"
    if variation_nr is not None:
        run_name = f"{run_name}_{variation_nr}"
    return ABLATION_NO_RANDOM_FEATURES_TRAIN_REQUEST.train(nr_cpu_threads=1, run_name=run_name)


def train_ablation_no_hidden(variation_nr=None):
    run_name = "ablation_no_hidden_features"
    if variation_nr is not None:
        run_name = f"{run_name}_{variation_nr}"
    return ABLATION_NO_HIDDEN_FEATURES_TRAIN_REQUEST.train(nr_cpu_threads=1, run_name=run_name)


def load_ablation_no_random_features(variation_nr=None):
    model_name = f"ablation-no-random-features model"
    checkpoint_path = ABLATION_NO_RANDOM_FEATURES_CHECKPOINT_PATH
    if variation_nr is not None:
        model_name = f"{model_name} variation {variation_nr}"
        checkpoint_path = get_model_variation_checkpoint_path(checkpoint_path, variation_nr)
    return _load_model(
        checkpoint_path,
        lambda: train_ablation_no_random_features(variation_nr),
        model_name)


def load_ablation_no_hidden_features(variation_nr=None):
    model_name = "ablation-no-hidden-features model"
    checkpoint_path = ABLATION_NO_HIDDEN_FEATURES_CHECKPOINT_PATH
    if variation_nr is not None:
        model_name = f"{model_name} variation {variation_nr}"
        checkpoint_path = get_model_variation_checkpoint_path(checkpoint_path, variation_nr)
    return _load_model(
        checkpoint_path,
        lambda: train_ablation_no_random_features(variation_nr),
        model_name)


def get_model_variation_checkpoint_path(original_checkpoint_path: Path, variation_nr):
    return (original_checkpoint_path.parent / f"{original_checkpoint_path.stem}_{variation_nr}"
            ).with_suffix(original_checkpoint_path.suffix)
