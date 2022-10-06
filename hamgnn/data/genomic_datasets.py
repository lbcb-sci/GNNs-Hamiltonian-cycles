from pathlib import Path
import torch
import torch_geometric as torch_g
import pytorch_lightning as torch_lightning

from hamgnn.data.DataModules import LIGHTNING_MODULE_REFERENCE_KEYWORD
from hamgnn.data.GraphDataset import GraphExample, GraphDataLoader
from OlcGraph import OlcGraph
import olc_graph_tools


def is_string_graph_valid(graph_path):
    if not graph_path.exists():
        return False
    try:
        olc_g = OlcGraph.from_gml(graph_path)
        return True
    except Exception:
        return False


def is_unique_ground_truth_cycle(olc_graph):
    ground_truth_chains, ground_truth_loops = olc_graph_tools.find_chains_and_loops(olc_graph)
    if len(ground_truth_chains) != 1 or len(ground_truth_loops) != 1:
        return False
    return True


def is_ground_truth_contig_large(olc_graph, min_contig_fraction=0.5):
    ground_truth_chains, ground_truth_loops = olc_graph_tools.find_chains_and_loops(olc_graph)
    ground_truth_contigs = sorted(ground_truth_chains + ground_truth_loops, key=len, reverse=True)
    if len(ground_truth_contigs) == 0:
        return False
    elif len(ground_truth_contigs[0]) < olc_graph.num_nodes * min_contig_fraction:
        return False
    return True


class StringGraphsDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_filepaths: list[Path]):
        self.filepaths = list_of_filepaths

    def __len__(self):
        return len(self.filepaths)

    def _get_graph(self, idx):
        return OlcGraph.from_gml(self.filepaths[idx])

    def _get_teacher_path(self, olc_graph):
        return None

    def __getitem__(self, idx):
        olc_graph = OlcGraph.from_gml(self.filepaths[idx])
        graph = torch_g.data.Data(num_nodes=olc_graph.num_nodes, edge_index=olc_graph.edge_index)
        teacher_path = self._get_teacher_path(olc_graph)
        graph_example = GraphExample(graph=graph, teacher_path=torch.tensor(teacher_path))
        return graph_example


class StringGraphWithLargestContigDataset(StringGraphsDataset):
    def _get_teacher_path(self, olc_graph):
        ground_truth_chains, ground_truth_loops = olc_graph_tools.find_chains_and_loops(olc_graph)
        return sorted(ground_truth_chains + ground_truth_loops, key=len, reverse=True)[0]


class StringGraphDatamodule(torch_lightning.LightningDataModule):
    def __init__(self, train_paths, val_paths, test_paths, train_batch_size, val_batch_size, test_batch_size=1, *args, **kwargs):
        self.train_paths, self.val_paths, self.test_paths = [
            [Path(p) for p in paths] for paths in [train_paths, val_paths, test_paths]
        ]
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        kwargs.pop(LIGHTNING_MODULE_REFERENCE_KEYWORD)
        super().__init__(*args, **kwargs)

    def prepare_data(self) -> None:
        super().prepare_data()
        print("Checking locally stored genome graphs...")
        for graph_path in self.train_paths + self.val_paths + self.test_paths:
           is_string_graph_valid(graph_path)
        for graph_path in self.train_paths + self.val_paths:
            olc_graph = OlcGraph.from_gml(graph_path)
            is_ground_truth_contig_large(olc_graph)

    def train_dataloader(self):
        return GraphDataLoader(StringGraphWithLargestContigDataset(self.train_paths), batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return GraphDataLoader(StringGraphWithLargestContigDataset(self.val_paths), batch_size=self.val_batch_size, shuffle=True)

    def test_dataloader(self):
        return GraphDataLoader(StringGraphsDataset(self.test_paths), batch_size=self.test_batch_size)
