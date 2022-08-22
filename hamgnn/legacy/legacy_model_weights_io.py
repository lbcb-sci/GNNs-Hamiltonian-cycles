import os
from pathlib import Path

import torch

from hamgnn.Models import EncodeProcessDecodeAlgorithm, GatedGCNEmbedAndProcess, WalkUpdater
from hamgnn.solution_scorers import CombinatorialScorer

LEGACY_WEIGHTS_STORAGE_DIR = "legacy_model_weights"

LEGACY_HAMS_CLASS = EncodeProcessDecodeAlgorithm
LEGACY_HAMS_CLASS_NAME = EncodeProcessDecodeAlgorithm.__name__
LEGACY_HAMS_CONSTRUCTOR_PARAMETERS = {"processor_depth": 5, "in_dim": 1, "out_dim": 1, "hidden_dim": 32, "graph_updater_class": WalkUpdater, "loss_type": "mse"}
LEGACY_HAMS_PROCESSOR_FILENAME = f"{LEGACY_HAMS_CLASS_NAME}_Processor.tar"
LEGACY_HAMS_ENCODER_FILENAME = f"{LEGACY_HAMS_CLASS_NAME}_Encoder.tar"
LEGACY_HAMS_DECODER_FILENAME = f"{LEGACY_HAMS_CLASS_NAME}_Decoder.tar"
LEGACY_HAMS_INITIAL_HIDDEN_TENSOR_FILENAME = f"{LEGACY_HAMS_CLASS_NAME}_InitialHiddenTensor.pt"

LEGACY_HAMR_CLASS = GatedGCNEmbedAndProcess
LEGACY_HAMR_CLASS_NAME = LEGACY_HAMR_CLASS.__name__
LEGACY_HAMR_EMBEDDING_FILENAME = f"{LEGACY_HAMR_CLASS_NAME}-Embedding.tar"
LEGACY_HAMR_PROCESSOR_FILENAME = f"{LEGACY_HAMR_CLASS_NAME}-Processor.tar"

LEGACY_HAMR_CONSTRUCTOR_PARAMETERS = {
    "in_dim": 3, "out_dim": 2, "hidden_dim": 32, "embedding_depth": 8, "processor_depth": 5, "value_function_weight": 1,
    "l2_regularization_weight": 0.01, "loss_type": "mse", "graph_updater_class": WalkUpdater, "solution_scorer": CombinatorialScorer()}
LEGACY_HAMR_EMBEDDING_DEPTH = 8
LEGACY_HAMR_PROCESSOR_DEPTH = 5
LEGACY_HAMR_HIDDEN_DIM = 32

def _get_HamS_weights_path(directory: Path):
    return {parameter_name: directory / parameter_filename for parameter_name, parameter_filename in zip(
        ["encoder_path", "decoder_path", "processor_path", "initial_hidden_path"],
        [LEGACY_HAMS_ENCODER_FILENAME, LEGACY_HAMS_DECODER_FILENAME, LEGACY_HAMS_PROCESSOR_FILENAME, LEGACY_HAMS_INITIAL_HIDDEN_TENSOR_FILENAME]
        )}

def _get_HamR_weights_path(directory: Path):
    return {parameter_name: directory / submodule_filename for parameter_name, submodule_filename in zip(
        ["embedding_path", "processor_path"],
        [LEGACY_HAMR_EMBEDDING_FILENAME, LEGACY_HAMR_PROCESSOR_FILENAME])}

def load_legacy_HamS(directory=LEGACY_WEIGHTS_STORAGE_DIR):
    HamS = LEGACY_HAMS_CLASS(**LEGACY_HAMS_CONSTRUCTOR_PARAMETERS)

    directory = Path(directory)
    paramteter_paths = _get_HamS_weights_path(directory)
    encoder_path, decoder_path, processor_path, initial_hidden_path = [paramteter_paths[tag] for tag in ["encoder_path", "decoder_path", "processor_path", "initial_hidden_path"]]
    for module, path in zip(
            [HamS.encoder_nn, HamS.decoder_nn, HamS.processor_nn],
            [encoder_path, decoder_path, processor_path]):
        if os.path.isfile(path):
            module.load_state_dict(torch.load(path))
            print("Loaded {} from {}".format(module.__class__.__name__, path))
        else:
            print(f"Failed to load parameters from {path}")
    if os.path.isfile(initial_hidden_path):
        HamS.initial_h = torch.nn.Parameter(torch.load(initial_hidden_path))
    else:
        print(f"Faliled to load parameters from {path}")
        HamS.initial_h = torch.nn.Parameter(torch.rand(HamS.hidden_dim, device=HamS.device))
    return HamS


def load_legacy_HamR(directory=LEGACY_WEIGHTS_STORAGE_DIR):
    HamR = LEGACY_HAMR_CLASS(**LEGACY_HAMR_CONSTRUCTOR_PARAMETERS)
    directory = Path(directory)
    submodules_filepath = _get_HamR_weights_path(directory)
    embedding_path, processor_path = [submodules_filepath[submodule_name] for submodule_name in ["embedding_path", "processor_path"]]

    if os.path.isfile(embedding_path):
        embedding_save_data = torch.load(embedding_path)
        HamR.initial_embedding = torch.nn.Parameter(embedding_save_data["initial"]["initial"])
        for module_key, module in zip(["MPNN", "out_projection"], [HamR.embedding, HamR.embedding_out_projection]):
            module.load_state_dict(embedding_save_data[module_key])
        print("Loaded embedding submodule from {}".format(embedding_path))
    else:
        print(f"Failed to load embedding submodule from {embedding_path}")
    if os.path.isfile(processor_path):
        processor_save_data = torch.load(processor_path)
        for modul_key, module in zip(["MPNN", "out_projection"], [HamR.processor, HamR.processor_out_projection]):
            module.load_state_dict(processor_save_data[modul_key])
        print("Loaded processor submodule from {}".format(processor_path))
    else:
        print(f"Failed to load processor submodule from {processor_path}")
    return HamR


def store_legacy_model(model, directory):
    directory = Path(directory)
    if isinstance(model, LEGACY_HAMS_CLASS):
        parameter_paths = _get_HamS_weights_path(directory)
        encoder_path, decoder_path, processor_path, initial_hidden_path = [parameter_paths[tag] for tag in ["encoder_path", "decoder_path", "processor_path", "initial_hidden_path"]]
        model = model
        for submodule_name, submodule, path in zip(
                ["processor", "encoder", "decoder"],
                [model.processor_nn, model.encoder_nn, model.decoder_nn],
                [processor_path, encoder_path, decoder_path]):
            torch.save(submodule.state_dict(), path)
            print(f"Saved {submodule_name} weights to {path}")
        torch.save(model.initial_h, initial_hidden_path)
        print(f"Saved initial hidden tensor to {initial_hidden_path}")
    elif isinstance(model, LEGACY_HAMR_CLASS):
        parameter_paths = _get_HamR_weights_path(directory)
        embedding_path, processor_path = [parameter_paths[tag] for tag in ["embedding_path", "processor_path"]]
        embedding_save_data = {
            'initial': {"initial": model.initial_embedding},
            'MPNN': model.embedding.state_dict(),
            'out_projection': model.embedding_out_projection.state_dict()
        }
        processor_save_data = {
            'MPNN': model.processor.state_dict(),
            'out_projection': model.processor_out_projection.state_dict()
        }
        for submodule_name, save_data, path in zip(["embedding", "processor"], [embedding_save_data, processor_save_data],
                                   [embedding_path, processor_path]):
            torch.save(save_data, path)
            print(f"Save {submodule_name} submodule weights to {path}")


def load_legacy_models(directory=LEGACY_WEIGHTS_STORAGE_DIR):
    return {"HamS": load_legacy_HamS(directory), "HamR": load_legacy_HamR(directory)}
