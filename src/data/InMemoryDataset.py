import os
import pickle
import itertools
from typing import List
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import torch.utils.data
import torch_geometric

import src.data.GraphGenerators as GraphGenerators
import src.data.GraphDataset as GraphDataset
import src.ExactSolvers as ExactSolvers
from src.data.GraphDataset import GraphExample


class ErdosRenyiGraphExample(GraphExample):
    def __init__(self, graph: torch_geometric.data.Data, hamiltonian_cycle: torch.Tensor, hamilton_existence_probability=None) -> None:
        super().__init__(graph, hamiltonian_cycle, None)
        self.hamiltonian_cycle = hamiltonian_cycle
        self.hamilton_existence_probability = hamilton_existence_probability


class ErdosRenyiInMemoryDataset(torch.utils.data.Dataset):
    STORAGE_EDGE_INDEX_TAG = "edge_index"
    STORAGE_HAMILTONIAN_CYCLE_TAG = "hamilton_cycle"
    STORAGE_NUM_NODES_TAG = "num_nodes"
    #TODO needs changing to hamilton_existance_probability
    STORAGE_HAMILTON_EXISTENCE_PROB = "prob"

    class Transforms:
        pass
        @staticmethod
        def graph_and_hamilton_cycle(item):
            graph = item[ErdosRenyiInMemoryDataset.STORAGE_EDGE_INDEX_TAG]
            cycle = torch.tensor(item[ErdosRenyiInMemoryDataset.STORAGE_HAMILTONIAN_CYCLE_TAG])
            return GraphDataset.GraphExample(graph, cycle)

    # @staticmethod
    # def _raw_to_storage_dict(graph, hamiltonian_cycle, hamilton_existence_prob=-1):
    #     edge_index = graph.edge_index
    #     edge_index_list = [list(edge_index[i]) for i in range(2)]
    #     storage_dict = {ErdosRenyiInMemoryDataset.STORAGE_EDGE_INDEX_TAG: edge_index_list,
    #             ErdosRenyiInMemoryDataset.STORAGE_NUM_NODES_TAG: graph.num_nodes,
    #             ErdosRenyiInMemoryDataset.STORAGE_HAMILTON_EXISTENCE_PROB: hamilton_existence_prob,
    #             ErdosRenyiInMemoryDataset.STORAGE_HAMILTON_CYCLE_TAG: hamiltonian_cycle}
    #     return storage_dict

    @staticmethod
    def to_storage_dict(graph_examples: List[ErdosRenyiGraphExample]):
        storage_dict = defaultdict(list)
        for ex in graph_examples:
            edge_index = ex.graph.edge_index
            edge_index_list = [[int(x.item()) for x in edge_index[i]] for i in range(2)]
            storage_dict[ErdosRenyiInMemoryDataset.STORAGE_EDGE_INDEX_TAG].append(edge_index_list)
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
                ErdosRenyiInMemoryDataset.STORAGE_NUM_NODES_TAG,
                ErdosRenyiInMemoryDataset.STORAGE_HAMILTONIAN_CYCLE_TAG,
                ErdosRenyiInMemoryDataset.STORAGE_HAMILTON_EXISTENCE_PROB]
              ])
        # a1, a2, a3, a4 = [storage_dict[k] for k in [
        #     ErdosRenyiInMemoryDataset.STORAGE_EDGE_INDEX_TAG,
        #     ErdosRenyiInMemoryDataset.STORAGE_NUM_NODES_TAG,
        #     ErdosRenyiInMemoryDataset.STORAGE_HAMILTONIAN_CYCLE_TAG,
        #     ErdosRenyiInMemoryDataset.STORAGE_HAMILTON_EXISTENCE_PROB]]
        # b = zip(a1, a2, a3, a4)
        # breakpoint()
        for edge_index, num_nodes, hamiltonian_cycle, hamilton_existence_probability in _zipped_dict:
            edge_index = torch.tensor(edge_index, device=device)
            graph = torch_geometric.data.Data(num_nodes=num_nodes, edge_index=edge_index)
            data_list.append(ErdosRenyiGraphExample(graph, torch.tensor(hamiltonian_cycle), hamilton_existence_probability))
        return data_list

    @staticmethod
    def create_dataset(out_folder, sizes, nr_examples_per_size=200, hamilton_existence_prob=0.8, solve_with_concorde=True):
        concorde = ExactSolvers.ConcordeHamiltonSolver()
        progress_bar = tqdm(sizes, desc=f"Creating graph datasets")
        for s in progress_bar:
            progress_bar.set_description(f"Creating dataset of graph of size {s}")
            data = []
            generator = GraphGenerators.ErdosRenyiGenerator(num_nodes=s, hamilton_existence_probability=hamilton_existence_prob)
            for g in tqdm(itertools.islice(generator, nr_examples_per_size), total=nr_examples_per_size, leave=False):
                hamiltonian_cycle = concorde.solve(g) if solve_with_concorde else None
                data.append(ErdosRenyiGraphExample(g, hamiltonian_cycle, hamilton_existence_prob))
            filepath = Path(out_folder) / "Erdos_Renyi({},{:05d}).pt".format(s, int(generator.p*10_000))
            ErdosRenyiInMemoryDataset.save_to_file(filepath, data)

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
