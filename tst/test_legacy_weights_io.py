from pathlib import Path

import torch

import src.legacy.legacy_model_weights_io as legacy_io

if __name__ == "__main__":
    models = legacy_io.load_legacy_models()
    output_dir = Path(__file__).parent / "legacy_io_test"
    output_dir.mkdir(exist_ok=True)
    for key, model in models.items():
        legacy_io.store_legacy_model(model, output_dir)
    reconstructed_models = legacy_io.load_legacy_models(output_dir)
    assert models.keys() == reconstructed_models.keys()
    for k in models.keys():
        model = models[k]
        reconstructed_model = reconstructed_models[k]
        assert model.hparams == reconstructed_model.hparams
        for parameter, reconstructed_parameter in zip(model.parameters(), reconstructed_model.parameters()):
            assert torch.isclose(parameter, reconstructed_parameter).all().item()
    print("Loaded and re-stored HamS and HamR models. Old and new weights match")