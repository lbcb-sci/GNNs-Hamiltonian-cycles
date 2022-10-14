from pathlib import Path

import torch.utils.data
import torch_geometric.data

from hamgnn.data.GraphDataset import GraphExample

class FHCPDataset(torch.utils.data.Dataset):
    def __init__(self, FHCP_dataset_directors: Path):
        self.dataset_directory = Path(FHCP_dataset_directors)
        self.graph_filepaths, self.solution_filepaths = [
            [p for p in self.dataset_directory.iterdir() if p.name.startswith("graph") and p.suffix == suffix]
            for suffix in [".hcp", ".tou"]]
        sort_key = lambda path: int(path.stem.split(".")[0].replace("graph", ""))
        self.graph_filepaths.sort(key=self._extract_problem_number_from_path)
        self.solution_filepaths.sort(key=sort_key)
        assert len(self.graph_filepaths) == len(self.solution_filepaths),\
            (f"FHCP dataset in {self.dataset_directory} seems to be corrupt. The number of graphs ({len(self.graph_filepaths)})"
             f" does not match the number of solutions ({len(self.solution_filepaths)})")
        for graph_path, solution_path in zip(self.graph_filepaths, self.solution_filepaths):
            assert solution_path.stem.startswith(graph_path.stem), \
            (f"Could not match graph files to solutions. {graph_path} does not match with {solution_path}")

    def _extract_problem_number_from_path(self, path):
        return int(path.stem.split(".")[0].replace("graph", ""))

    def load_graph(self, graph_path, solution_path):
        edges_set = set()
        dimension = None
        with graph_path.open("r") as graph_file:
            _is_edges_flag = False
            for line in graph_file:
                l = line.strip()
                if _is_edges_flag:
                    if l == "-1":
                        break
                    edges_set.add(tuple(int(x) for x in l.split()))
                elif l.startswith("DIMENSION"):
                    dimension = int(l.split()[-1])
                elif line.strip() == "EDGE_DATA_SECTION":
                    _is_edges_flag = True
        assert dimension is not None, f"Failed to find the dimension of graph in {graph_path}"
        edges = set((start, end) for (start, end) in edges_set if start < end)
        edges_index = torch.tensor([e for e in edges]).t() - 1 # Different node indexing convention
        edges_index = torch.cat([edges_index, edges_index.flip(0)], dim=-1)

        tour = []
        with solution_path.open("r") as solution_file:
            _is_tour_flag = False
            for line in solution_file:
                l = line.strip()
                if _is_tour_flag:
                    if l == "-1":
                        break
                    tour.append(int(l))
                elif l == "TOUR_SECTION":
                    _is_tour_flag = True
        tour.append(tour[0]) # Different cycle convention
        tour = torch.tensor(tour) - 1 # Different node indexing convention
        graph = torch_geometric.data.Data(num_nodes=dimension, edge_index=edges_index)
        return GraphExample(graph, tour)

    def get_problem_description(self, index):
        path = self.graph_filepaths[index]
        return path.name, self._extract_problem_number_from_path(path)

    def __len__(self):
        return len(self.graph_filepaths)

    def __getitem__(self, index):
        return self.load_graph(self.graph_filepaths[index], self.solution_filepaths[index])
