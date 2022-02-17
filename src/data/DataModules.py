import pytorch_lightning as torch_lightning

import src.constants as constants
from src.data.InMemoryDataset import ErdosRenyiInMemoryDataset
from src.data.GraphDataset import GraphGeneratingDataset, GraphDataLoader, SimulationStatesDataLoader, SimulationsDataset
from src.data.GraphGenerators import ErdosRenyiGenerator, NoisyCycleGenerator


SIMULATION_MODULE_VARIABLE_NAME = "simulation_module_reference"


class BaseGraphGeneratingDataModule(torch_lightning.LightningDataModule):
    def __init__(self, *args, **kwargs):
        self.train_graph_size = kwargs.pop("train_graph_size", 25)
        self.train_batch_size = kwargs.pop("train_batch_size", 8)
        self.train_virtual_epoch_size = kwargs.pop("train_virtual_epoch_size", 2000)
        self.val_graph_size = kwargs.pop("val_graph_size", 25)
        self.val_batch_size = kwargs.pop("val_batch_size", 8)
        self.val_virtual_epoch_size = kwargs.pop("val_virtual_epoch_size", 100)

        super().__init__(*args, **kwargs)


class TestWithLocalDatasetDataModule(BaseGraphGeneratingDataModule):
    def __init__(self, test_data_directories=None, test_batch_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if test_data_directories is None:
            self.test_data_directories = constants.EVALUATION_MINIMAL_TEST_DATA_FOLDERS
        else:
            self.test_data_directories = test_data_directories
        self.test_batch_size = test_batch_size

    def prepare_data(self) -> None:
        return super().prepare_data()

    def test_dataloader(self):
        return GraphDataLoader(ErdosRenyiInMemoryDataset(self.test_data_directories), batch_size=self.test_batch_size)


class ArtificialCycleDataModule(TestWithLocalDatasetDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_noise_prob_per_edge = kwargs.pop("train_noise_prob_per_edge", 0.8)
        self.val_noise_prob_per_edge = kwargs.pop("val_noise_prob_per_edge", 0.8)

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage=None) -> None:
        return super().setup(stage)

    def train_dataloader(self):
        generator = NoisyCycleGenerator(self.train_graph_size, self.train_noise_prob_per_edge)
        return GraphDataLoader(GraphGeneratingDataset(generator, self.train_virtual_epoch_size), batch_size=self.train_batch_size)

    def val_dataloader(self):
        generator = NoisyCycleGenerator(self.val_graph_size, self.val_noise_prob_per_edge)
        return GraphDataLoader(GraphGeneratingDataset(generator, self.val_virtual_epoch_size), batch_size=self.val_batch_size)

class ReinforcementErdosRenyiDataModule(TestWithLocalDatasetDataModule):
    def __init__(self, *args, **kwargs):
        simulation_module =  kwargs.pop(SIMULATION_MODULE_VARIABLE_NAME)
        super().__init__(*args, **kwargs)

        self.train_ham_existence_prob = kwargs.pop("train_ham_existence_prob", 0.8)
        self.val_ham_existence_prob = kwargs.pop("val_ham_existence_prob", 0.8)
        self.train_nr_simultaneous_simulations = kwargs.pop("train_nr_simultaneous_simulations", 4)
        self.val_nr_simultaneous_simulations = kwargs.pop("val_nr_simultaneous_simulations", 1)
        self.simulation_module = simulation_module

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage = None) -> None:
        return super().setup(stage)

    def train_dataloader(self):
        generator = ErdosRenyiGenerator(self.train_graph_size, self.train_ham_existence_prob)
        return SimulationStatesDataLoader(
            SimulationsDataset(generator, self.simulation_module, self.train_virtual_epoch_size, nr_simultaneous_simulations=self.train_nr_simultaneous_simulations),
            batch_size=self.train_batch_size)

    def val_dataloader(self):
        generator = ErdosRenyiGenerator(self.val_graph_size, self.val_ham_existence_prob)
        return SimulationStatesDataLoader(
            SimulationsDataset(generator, self.simulation_module, self.val_virtual_epoch_size, nr_simultaneous_simulations=self.val_nr_simultaneous_simulations),
            batch_size=self.val_batch_size)
