import os
import pickle
import itertools

import torch.utils.data

import src.data.GraphGenerators as GraphGenerators
import src.data.GraphDataset as GraphDataset
import src.ExactSolvers as ExactSolvers

class ErdosRenyiInMemoryDataset(torch.utils.data.Dataset):
    GRAPH_TAG = "graph"
    HAMILTON_CYCLE_TAG = "hamilton_cycle"
    NUM_NODES_TAG = "num_nodes"
    ERDOS_RENYI_PROB_TAG = "prob"

    class Transforms:
        @staticmethod
        def graph_and_hamilton_cycle(item):
            graph = item[ErdosRenyiInMemoryDataset.GRAPH_TAG]
            cycle = torch.tensor(item[ErdosRenyiInMemoryDataset.HAMILTON_CYCLE_TAG])
            return GraphDataset.GraphExample(graph, cycle)

    @staticmethod
    def create_dataset(out_folder, sizes, nr_examples_per_size=200, hamilton_existence_prob=0.8, number_of_threads=32,
                       solve_with_concorde=True):
        torch.set_num_threads(number_of_threads)
        concorde = ExactSolvers.ConcordeHamiltonSolver()
        file_index = 1
        for s in sizes:
            data_list = []
            generator = GraphGenerators.ErdosRenyiGenerator(num_nodes=s, hamilton_existence_probability=hamilton_existence_prob)
            for g in itertools.islice(generator, nr_examples_per_size):
                hamilton_cycle = concorde.solve(g) if solve_with_concorde else None
                item = {ErdosRenyiInMemoryDataset.GRAPH_TAG: g,
                        ErdosRenyiInMemoryDataset.NUM_NODES_TAG: generator.num_nodes,
                        ErdosRenyiInMemoryDataset.ERDOS_RENYI_PROB_TAG: generator.p,
                        ErdosRenyiInMemoryDataset.HAMILTON_CYCLE_TAG: hamilton_cycle}
                data_list += [item]
            filename = "Erdos_Renyi({},{:05d}).pt".format(s, int(generator.p*10_000))
            print(f"Creating {filename} ({file_index}/{len(sizes)})...")
            file_index += 1
            with open(os.path.join(out_folder, filename), 'wb') as out:
                pickle.dump(data_list, out)

    def __init__(self, path_list, transform=Transforms.graph_and_hamilton_cycle):
        assert path_list is not None
        self.transform = transform
        self.data_list = []
        search_tree = []
        for path in path_list:
            to_check = [os.path.join(path, f) for f in os.listdir(path)] if os.path.isdir(path) else [path]
            search_tree += [f for f in to_check if os.path.isfile(f) and os.path.splitext(f)[1] == ".pt"]
        for path in search_tree:
            with open(path, 'rb') as in_file:
                self.data_list += pickle.load(in_file)

    def __len__(self):
        return self.data_list.__len__()

    def __getitem__(self, idx):
        item = self.data_list[idx]
        if self.transform is not None:
            item = self.transform(item)
        return item
