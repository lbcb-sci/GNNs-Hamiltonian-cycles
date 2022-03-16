import sys

import wandb

import src.constants as constants
import src.Models as Models
import src.model_utils as model_utils

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        print("Please provide weights and biases id of the model to train")

    wandb_id = args[-1]
    wandb_project = constants.WEIGHTS_AND_BIASES_PROJECT
    wandb_kwargs = {"project": wandb_project, "resume": True}
    wandb_run = wandb.init(project=wandb_project, id=wandb_id, resume=True)
    model = model_utils.create_model_for_wandb_run(wandb_run, wandb_run["checkpoint"])

    results = model_utils.test_on_saved_data(model, wandb_run=wandb_run)
    wandb_run.finish()
    print(results)
