import pytorch_lightning as torch_lightning

import src.constants as constants
from src.data.InMemoryDataset import ErdosRenyiInMemoryDataset
from src.data.GraphDataset import GraphGeneratingDataset, GraphDataLoader, SimulationStatesDataLoader, SimulationsDataset
from src.data.GraphGenerators import ErdosRenyiGenerator, NoisyCycleGenerator


LIGHTNING_MODULE_REFERENCE = "lightning_module_reference"


class BaseGraphGeneratingDataModule(torch_lightning.LightningDataModule):
    def __init__(self, train_graph_size=25, train_batch_size=8, train_virtual_epoch_size=2000, val_graph_size=None, val_batch_size=None,
                 val_virtual_epoch_size=None, *args, **kwargs):
        self.__lightnig_module_reference =  kwargs.pop(LIGHTNING_MODULE_REFERENCE)
        self.train_graph_size = train_graph_size
        self.train_batch_size = train_batch_size
        self.train_virtual_epoch_size = train_virtual_epoch_size
        self.val_graph_size = val_graph_size if val_graph_size else self.train_graph_size
        self.val_batch_size = val_batch_size if val_batch_size else self.train_batch_size
        self.val_virtual_epoch_size = val_virtual_epoch_size if val_virtual_epoch_size else self.train_virtual_epoch_size

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
    def __init__(self, train_noise_prob_per_edge=0.1, val_noise_prob_per_edge=None, *args, **kwargs):
        self.train_noise_prob_per_edge = train_noise_prob_per_edge
        self.val_noise_prob_per_edge = val_noise_prob_per_edge if val_noise_prob_per_edge else self.train_noise_prob_per_edge
        super().__init__(*args, **kwargs)

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
    def __init__(self, train_ham_existence_prob=0.8, val_ham_existence_prob=None, train_nr_simultaneous_simulations=4,
                 val_nr_simultaneous_simulations=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulation_module = self.__lightnig_module_reference
        self.train_ham_existence_prob = train_ham_existence_prob
        self.val_ham_existence_prob = val_ham_existence_prob if val_ham_existence_prob else self.train_ham_existence_prob
        self.train_nr_simultaneous_simulations = train_nr_simultaneous_simulations
        self.val_nr_simultaneous_simulations = val_nr_simultaneous_simulations

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
