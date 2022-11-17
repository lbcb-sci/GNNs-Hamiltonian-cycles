from argparse import ArgumentParser
import sys

import hamgnn.models_list as models_list
import hamgnn.model_utils as model_utils


if __name__ == "__main__":
    parser = ArgumentParser(f"Trains one of the models defined in {models_list.__file__} specified by a name")
    parser.add_argument("target", type=str, help="Name of the class of model to train")
    parser.add_argument("--run-name", type=str, default=None, help="Gives a name to wandb run for this training experiment")
    parser.add_argument("--gpu", type=int, default=None, help="Overrides gpu on which to train")
    args = parser.parse_args()
    target_name = args.target
    run_name = args.run_name
    gpu_id_override = args.gpu

    possible_targets = [var_name for var_name in dir(models_list) if isinstance(getattr(models_list, var_name, None), model_utils.ModelTrainRequest)]
    if target_name in possible_targets:
        train_request = getattr(models_list, target_name)
        if gpu_id_override is not None:
            train_request.arguments["trainer_hyperparams"]["gpus"] = [gpu_id_override]

        train_request.train(nr_cpu_threads=1, run_name=run_name)
    else:
        print(f"Could not find model request {target_name} to train. Please use one of the following as an argument")
        for possible_target in possible_targets:
            print(f'"{possible_target}"')
