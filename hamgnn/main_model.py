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


def store_as_main_model(checkpoint_path):
    shutil.copyfile(checkpoint_path, MAIN_MODEL_CHECKPOINT_PATH)


def train_main_model():
    return MAIN_MODEL_TRAIN_REQUEST.train(nr_cpu_threads=1, run_name="main_model")


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
    return _load_model(MAIN_MODEL_CHECKPOINT_PATH, train_main_model, "main model")


def train_ablation_no_random_features():
    return ABLATION_NO_RANDOM_FEATURES_TRAIN_REQUEST.train(nr_cpu_threads=1, run_name="ablation_no_random_features")


def train_ablation_no_hidden():
    return ABLATION_NO_HIDDEN_FEATURES_TRAIN_REQUEST.train(nr_cpu_threads=1, run_name="ablation_no_hidden_features")


def load_ablation_no_random_features():
    return _load_model(ABLATION_NO_RANDOM_FEATURES_CHECKPOINT_PATH, train_ablation_no_random_features, "ablation-no-random-features model")


def load_ablation_no_hidden_features():
    return _load_model(ABLATION_NO_HIDDEN_FEATURES_CHECKPOINT_PATH, train_ablation_no_hidden, "ablation-no-hidden-features model")
