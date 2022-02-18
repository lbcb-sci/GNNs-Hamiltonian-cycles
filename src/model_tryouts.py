import src.Models as Models
import src.data.DataModules as DataModules
import src.constants as constants
import src.train_model as train_model

class ModelTrainRequest:
    def __init__(self, **kwargs):
        self.arguments = kwargs

    def train(self):
        train_model.train_model(**self.arguments)


train_request_HamS_model = ModelTrainRequest(
    model_class = Models.EncodeProcessDecodeAlgorithm,
    datamodule_class = DataModules.ArtificialCycleDataModule,
    model_checkpoint = None,
    model_hyperparams = {
        "processor_depth":5,
        "loss_type": "entropy",
    },
    trainer_hyperparams = {
        "max_epochs": 100,
        "num_sanity_val_steps": 2,
        "log_every_n_steps": 50,

    },
    datamodule_hyperparams = {
        "train_virtual_epoch_size": 32 * 100,
        "train_batch_size": 32,
        "train_graph_size": 25,
        "train_noise_prob_per_edge": 1/25,
    }
)


if __name__ == "__main__":
    train_request_HamS_model.train()