import torch
import torch_geometric as torch_geometric
from typing import List


class GraphExample:
    def __init__(self, graph:torch_geometric.data.Data, teacher_path: torch.Tensor, teacher_distribution: torch.Tensor=None) -> None:
        self.graph = graph
        self.teacher_path = teacher_path
        self.teacher_distribution = teacher_distribution

class GraphBatchExample:
    def __init__(self, graph_batch: torch_geometric.data.Batch, teacher_paths: List[torch.Tensor], teacher_distributions: torch.Tensor=None) -> None:
        self.graph_batch = graph_batch
        self.teacher_paths = teacher_paths
        self.teacher_distributions = teacher_distributions


class GraphGeneratingDataset(torch.utils.data.Dataset):
    def __init__(self, graph_generator, virtual_epoch_size=1000, ) -> None:
        super().__init__()
        self.graph_generator = iter(graph_generator)
        self.virtual_epoch_size = virtual_epoch_size

    def __len__(self):
        return self.virtual_epoch_size

    def __getitem__(self, idx):
        generated_graph_example = next(self.graph_generator)
        return generated_graph_example


class GraphDataLoader(torch.utils.data.DataLoader):
    @staticmethod
    def graph_collate_fn(graph_examples: List[GraphExample]) -> GraphBatchExample:
        graphs = [example.graph for example in graph_examples]
        teacher_paths = [example.teacher_path for example in graph_examples]
        graph_sizes = [graph.num_nodes for graph in graphs]
        batched_data = torch_geometric.data.Batch.from_data_list(graphs)
        batch_shift = 0
        for index_of_walk, walk in enumerate(teacher_paths):
            walk += batch_shift
            batch_shift += graph_sizes[index_of_walk]
        graph_batch_example = GraphBatchExample(batched_data, teacher_paths)
        # TODO see if pytorch_lighting can be adjusted to move custom classes from and to devices.
        # It handles dictionaries of tensors without problems
        return {
            "num_nodes": graph_batch_example.graph_batch.num_nodes,
            "edge_index": graph_batch_example.graph_batch.edge_index,
            "batch_vector": graph_batch_example.graph_batch.batch,
            "teacher_paths": graph_batch_example.teacher_paths,
        }

    def __init__(self, dataset: GraphGeneratingDataset, *args, **kwargs) -> None:
        super().__init__(dataset, collate_fn=GraphDataLoader.graph_collate_fn, *args, **kwargs)
