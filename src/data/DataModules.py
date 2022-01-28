import pytorch_lightning as torch_lightning

import src.constants as constants
from src.data.InMemoryDataset import ErdosRenyiInMemoryDataset
from src.data.GraphDataset import GraphGeneratingDataset, GraphDataLoader
from src.data.GraphGenerators import NoisyCycleGenerator

class TestWithLocalDatasetDataLoader(torch_lightning.LightningDataModule):
    def __init__(self, test_data_directories=None, test_batch_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if test_data_directories is None:
            self.test_data_directories = constants.EVALUATION_DATA_FOLDERS
        else:
            self.test_data_directories = test_data_directories
        self.test_batch_size = test_batch_size

    def prepare_data(self) -> None:
        return super().prepare_data()

    def test_dataloader(self):
        return GraphDataLoader(ErdosRenyiInMemoryDataset(self.test_data_directories), batch_size=self.test_batch_size)

class ArtificialCycleDataModule(TestWithLocalDatasetDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_noise_prob_per_edge = 0.8
        self.train_graph_size = 25
        self.train_batch_size = 8
        self.val_noise_prob_per_edge = 0.8
        self.val_graph_size = 25
        self.val_batch_size = 8

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage=None) -> None:
        return super().setup(stage)

    def train_dataloader(self):
        generator = NoisyCycleGenerator(self.train_graph_size, self.train_noise_prob_per_edge)
        return GraphDataLoader(GraphGeneratingDataset(generator, 700), batch_size=self.train_batch_size)

    def val_dataloader(self):
        generator = NoisyCycleGenerator(self.val_graph_size, self.val_noise_prob_per_edge)
        return GraphDataLoader(GraphGeneratingDataset(generator, 150), batch_size=self.val_batch_size)
