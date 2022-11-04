import shutil

import hamgnn.constants as constants
import hamgnn.model_utils as model_utils
import hamgnn.models_list as models_list

MAIN_MODEL_CHECKPOINT_PATH = constants.MAIN_MODEL_CHECKPOINT_PATH
MAIN_MODEL_CHECKPOINT_NAME = constants.MAIN_MODEL_CHECKPOINT_PATH.name
MAIN_MODEL_TRAIN_REQUEST = models_list.train_request_HamS_gpu_with_rand_node_encoding


def store_as_main_model(checkpoint_path):
    shutil.copyfile(checkpoint_path, MAIN_MODEL_CHECKPOINT_PATH)


def train_main_model():
    return MAIN_MODEL_TRAIN_REQUEST.train(1, "main_model")


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


def load_main_model():
    if MAIN_MODEL_CHECKPOINT_PATH.exists():
        model, load_message = model_utils.load_existing_model(MAIN_MODEL_CHECKPOINT_PATH)
    else:
        is_train = _yes_no_input("The main model does not exist yet. Do you wish to train it?")
        if is_train:
            model, checkpoint_path = train_main_model()
            shutil.copyfile(checkpoint_path, MAIN_MODEL_CHECKPOINT_PATH)
    return model
