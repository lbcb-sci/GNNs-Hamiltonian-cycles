import sys

import src.models_list as models_list
import src.model_utils as model_utils


if __name__ == "__main__":
    possible_targets = [var_name for var_name in dir(models_list) if isinstance(getattr(models_list, var_name, None), model_utils.ModelTrainRequest)]
    args = sys.argv

    run_name = None
    assert len(args) <= 3
    if len(args) == 2:
        target_name = args[-1]
    if len(args) == 3:
        target_name = args[-2]
        run_name = args[-1]
    else:
        target_name = "train_request_HamS_larger_graphs"

    if target_name in possible_targets:
        getattr(models_list, target_name).train(run_name=run_name)
    else:
        print(f"Could not find model request {target_name} to train. Please use one of the following as an argument")
        for possible_target in possible_targets:
            print(f'"{possible_target}"')
