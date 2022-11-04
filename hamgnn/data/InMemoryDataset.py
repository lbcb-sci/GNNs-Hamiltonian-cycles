import pickle
import itertools
from typing import List, Callable
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from collections.abc import Iterable
import os
import sys

import torch.utils.data
import torch_geometric

import hamgnn.data.GraphGenerators as GraphGenerators
import hamgnn.data.GraphDataset as GraphDataset
import hamgnn.ExactSolvers as ExactSolvers
from hamgnn.data.GraphDataset import GraphExample


class ErdosRenyiGraphExample(GraphExample):
    def __init__(self, graph: torch_geometric.data.Data, edge_inclusion_probability, hamiltonian_cycle: torch.Tensor, hamilton_existence_probability=None) -> None:
        super().__init__(graph, hamiltonian_cycle, None)
        self.edge_inclusion_probability = edge_inclusion_probability
        self.hamiltonian_cycle = hamiltonian_cycle
        self.hamilton_existence_probability = hamilton_existence_probability


class ErdosRenyiInMemoryDataset(torch.utils.data.Dataset):
    STORAGE_EDGE_INDEX_TAG = "edge_index"
    STORAGE_EDGE_INCLUSION_PROBABILITY = "edge_inclusion_probability"
    STORAGE_HAMILTONIAN_CYCLE_TAG = "hamilton_cycle"
    STORAGE_NUM_NODES_TAG = "num_nodes"
    STORAGE_HAMILTON_EXISTENCE_PROB = "hamiltonian_cycle_probability"

    class Transforms:
        @staticmethod
        def graph_and_hamilton_cycle(item):
            graph = item[ErdosRenyiInMemoryDataset.STORAGE_EDGE_INDEX_TAG]
            cycle = torch.tensor(item[ErdosRenyiInMemoryDataset.STORAGE_HAMILTONIAN_CYCLE_TAG])
            return GraphDataset.GraphExample(graph, cycle)

    @staticmethod
    def to_storage_dict(graph_examples: List[ErdosRenyiGraphExample]):
        storage_dict = defaultdict(list)
        for ex in graph_examples:
            edge_index = ex.graph.edge_index
            edge_index_list = [[int(x.item()) for x in edge_index[i]] for i in range(2)]
            storage_dict[ErdosRenyiInMemoryDataset.STORAGE_EDGE_INDEX_TAG].append(edge_index_list)
            storage_dict[ErdosRenyiInMemoryDataset.STORAGE_EDGE_INCLUSION_PROBABILITY].append(ex.edge_inclusion_probability)
            storage_dict[ErdosRenyiInMemoryDataset.STORAGE_NUM_NODES_TAG].append(ex.graph.num_nodes)
            storage_dict[ErdosRenyiInMemoryDataset.STORAGE_HAMILTON_EXISTENCE_PROB].append(ex.hamilton_existence_probability),
            storage_dict[ErdosRenyiInMemoryDataset.STORAGE_HAMILTONIAN_CYCLE_TAG].append(ex.hamiltonian_cycle)
        return storage_dict

    @staticmethod
    def save_to_file(filepath: Path, data: List[ErdosRenyiGraphExample]):
        storage_dict = ErdosRenyiInMemoryDataset.to_storage_dict(data)
        with open(filepath, 'wb') as f:
            pickle.dump(storage_dict, f)

    @staticmethod
    def load_from_file(filepath: Path):
        with open(filepath, "rb") as f:
            storage_dict = pickle.load(f)
        data_list = ErdosRenyiInMemoryDataset.from_storage_dict(storage_dict)
        return data_list

    @staticmethod
    def from_storage_dict(storage_dict, device="cpu"):
        data_list = []
        _zipped_dict = zip(
            *[storage_dict[tag] for tag in [
                ErdosRenyiInMemoryDataset.STORAGE_EDGE_INDEX_TAG,
                ErdosRenyiInMemoryDataset.STORAGE_EDGE_INCLUSION_PROBABILITY,
                ErdosRenyiInMemoryDataset.STORAGE_NUM_NODES_TAG,
                ErdosRenyiInMemoryDataset.STORAGE_HAMILTONIAN_CYCLE_TAG,
                ErdosRenyiInMemoryDataset.STORAGE_HAMILTON_EXISTENCE_PROB]
              ])
        for edge_index, edge_inclusion_probability, num_nodes, hamiltonian_cycle, hamilton_existence_probability in _zipped_dict:
            edge_index = torch.tensor(edge_index, device=device)
            graph = torch_geometric.data.Data(num_nodes=num_nodes, edge_index=edge_index)
            data_list.append(ErdosRenyiGraphExample(graph, edge_inclusion_probability, torch.tensor(hamiltonian_cycle), hamilton_existence_probability))
        return data_list

    @staticmethod
    def _create_dataset_from_generator(out_folder,
                                           fn_get_generator:Callable[[int, float], GraphGenerators.ErdosRenyiGenerator],
                                           generator_first_params, nr_examples, generator_second_params, solve_with_concorde, is_show_progress):
        if not isinstance(nr_examples, Iterable):
            nr_examples = [nr_examples for _ in generator_first_params]
        if not isinstance(generator_second_params, Iterable):
            generator_second_params = [generator_second_params for _ in generator_first_params]

        concorde = ExactSolvers.ConcordeHamiltonSolver()
        with open(os.devnull, "w") as devnull_file:
            _progress_file = sys.stdout if is_show_progress else devnull_file
            progress_bar = tqdm(list(zip(generator_first_params, nr_examples, generator_second_params)), desc=f"Creating graph datasets", file=_progress_file)
            for s, nr_examples, ham_prob in progress_bar:
                progress_bar.set_description(f"Creating dataset with size={s} and prob {ham_prob}")
                data = []
                generator = fn_get_generator(s, ham_prob)
                edge_inclusion_probability = generator.p
                for g in tqdm(itertools.islice(generator, nr_examples), total=nr_examples, leave=False, file=_progress_file):
                    hamiltonian_cycle = concorde.solve(g) if solve_with_concorde else None
                    data.append(ErdosRenyiGraphExample(g, edge_inclusion_probability, hamiltonian_cycle, generator_second_params))
                filepath = Path(out_folder) / "Erdos_Renyi({},{:05d}).pt".format(s, int(edge_inclusion_probability*10_000))
                ErdosRenyiInMemoryDataset.save_to_file(filepath, data)

    @staticmethod
    def create_dataset_in_critical_regime(out_folder, sizes, nr_examples=200, hamilton_existence_prob=0.8, solve_with_concorde=True, is_show_progress=True):
        ErdosRenyiInMemoryDataset._create_dataset_from_generator(
            out_folder, lambda s, p: GraphGenerators.ErdosRenyiGenerator(s, p), sizes, nr_examples, hamilton_existence_prob, solve_with_concorde, is_show_progress
            )

    @staticmethod
    def create_dataset_from_edge_probabilities(out_folder, sizes, nr_examples, edge_existence_probability, solve_with_concorde=True, is_show_progress=True):
        ErdosRenyiInMemoryDataset._create_dataset_from_generator(
            out_folder, lambda s, p: GraphGenerators.ErdosRenyiGenerator.create_from_edge_probability(s, p), sizes, nr_examples,
            edge_existence_probability, solve_with_concorde, is_show_progress
            )

    def __init__(self, path_list, transform=None):
        assert path_list is not None
        self.transform = transform
        self.data_list = []
        search_tree = []
        path_list = [Path(p) for p in path_list]
        for path in path_list:
            to_check = [f for f in path.iterdir()] if path.is_dir() else [path]
            search_tree += [f for f in to_check if f.is_file() and f.suffix == ".pt"]
        self.data_list = []
        for path in search_tree:
            self.data_list += self.load_from_file(path)

    def __len__(self):
        return self.data_list.__len__()

    def __getitem__(self, idx):
        item = self.data_list[idx]
        if self.transform is not None:
            item = self.transform(item)
        return item
