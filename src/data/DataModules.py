import pytorch_lightning as torch_lightning

import src.constants as constants
from src.data.InMemoryDataset import ErdosRenyiInMemoryDataset
from src.data.GraphDataset import FilterSolvableGraphsGeneratingDataset, GraphGeneratingDataset, GraphDataLoader, SimulationStatesDataLoader, SimulationsDataset
from src.data.GraphGenerators import ErdosRenyiExamplesGenerator, ErdosRenyiGenerator, NoisyCycleGenerator


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
    def __init__(self, train_expected_noise_edges_per_node, val_expected_noise_edges_per_node=None, *args, **kwargs):
        self.train_expected_noise_edges_per_node = train_expected_noise_edges_per_node
        self.val_expected_noise_edges_per_node = val_expected_noise_edges_per_node if val_expected_noise_edges_per_node else self.train_expected_noise_edges_per_node
        super().__init__(*args, **kwargs)

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage=None) -> None:
        return super().setup(stage)

    def train_dataloader(self):
        generator = NoisyCycleGenerator(self.train_graph_size, self.train_expected_noise_edges_per_node)
        return GraphDataLoader(GraphGeneratingDataset(generator, self.train_virtual_epoch_size), batch_size=self.train_batch_size)

    def val_dataloader(self):
        generator = NoisyCycleGenerator(self.val_graph_size, self.val_expected_noise_edges_per_node)
        return GraphDataLoader(GraphGeneratingDataset(generator, self.val_virtual_epoch_size), batch_size=self.val_batch_size)


class ArtificialCycleWithDoubleEvaluationDataModule(ArtificialCycleDataModule):
    def __init__(self, train_expected_noise_edges_per_node, val_hamiltonian_existence_probability, val_expected_noise_edges_per_node=None, *args, **kwargs):
        self.val_hamiltonian_existence_probabilty = val_hamiltonian_existence_probability
        super().__init__(train_expected_noise_edges_per_node, val_expected_noise_edges_per_node, *args, **kwargs)

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage=None) -> None:
        return super().setup(stage)

    def val_dataloader(self):
        dataloader_like_train = super().val_dataloader()
        generator_like_test = ErdosRenyiExamplesGenerator(self.val_graph_size, self.val_hamiltonian_existence_probabilty)
        dataloader_like_test = GraphDataLoader(FilterSolvableGraphsGeneratingDataset(generator_like_test, self.val_virtual_epoch_size), batch_size=self.val_batch_size)
        return dataloader_like_train, dataloader_like_test

class SolvedErdosRenyiDataModule(TestWithLocalDatasetDataModule):
    def __init__(self, train_hamilton_existence_probability, val_hamilton_existence_probability=None, *args, **kwargs):
        self.train_hamilton_existence_probability = train_hamilton_existence_probability
        self.val_hamilton_existence_probability = val_hamilton_existence_probability if val_hamilton_existence_probability else train_hamilton_existence_probability
        super().__init__(*args, **kwargs)

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage=None) -> None:
        return super().setup(stage)

    def train_dataloader(self):
        generator = ErdosRenyiExamplesGenerator(self.train_graph_size, self.train_hamilton_existence_probability)
        return GraphDataLoader(FilterSolvableGraphsGeneratingDataset(generator, self.train_virtual_epoch_size), batch_size=self.train_batch_size)

    def val_dataloader(self):
        generator = ErdosRenyiExamplesGenerator(self.val_graph_size, self.val_hamilton_existence_probability)
        return GraphDataLoader(FilterSolvableGraphsGeneratingDataset(generator, self.val_virtual_epoch_size), batch_size=self.val_batch_size)


class ReinforcementErdosRenyiDataModule(TestWithLocalDatasetDataModule):
    def __init__(self, train_ham_existence_prob=0.8, val_ham_existence_prob=None, train_nr_simultaneous_simulations=4,
                 val_nr_simultaneous_simulations=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulation_module = kwargs[LIGHTNING_MODULE_REFERENCE]
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
