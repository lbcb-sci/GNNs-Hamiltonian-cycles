import torch
import torch_geometric as torch_g
from abc import ABC, abstractmethod
import time
from tqdm import tqdm


class HamiltonSolver(ABC):
    @abstractmethod
    def solve_graphs(self, graphs: list[torch_g.data.Data]) -> list[list[int]]:
        pass

    def solve(self, graph: torch_g.data.Data):
        return self.solve_graphs([graph])[0]

    def timed_solve_graphs(self, graphs: list[torch_g.data.Data], is_show_progress=False) -> list[list[int]]:
        times = []
        solutions = []
        if is_show_progress:
            graphs = tqdm(list(graphs))
        for g in graphs:
            g.to(self.device)
            start = time.thread_time_ns()
            solution = self.solve(g)
            end = time.thread_time_ns()
            solutions.append(solution)
            times.append((end - start) / 1000_000_000)
        return solutions, times


class DataUtils:
    @staticmethod
    def _current_indices(d: torch_g.data.Data):
        return torch.nonzero(torch.isclose(d.x[..., 1], torch.ones_like(d.x[..., 1]))).squeeze(-1)

    @staticmethod
    def _starting_indices(d: torch_g.data.Data):
        return torch.nonzero(torch.isclose(d.x[..., 0], torch.ones_like(d.x[..., 0]))).squeeze(-1)

    @staticmethod
    def _neighbor_indices(d: torch_g.data.Data):
        current = DataUtils._current_indices(d)
        if current.numel == 0:
            return []
        neighbor_index = d.edge_index[1, torch.any(d.edge_index[None, 0, :] == current[:, None], dim=0)]
        return neighbor_index.unique()
    @staticmethod
    def _mask_neighbor_logits(logits, d: torch_g.data.Data):
        valid_next_step_indices = torch.cat([DataUtils._neighbor_indices(d), DataUtils._current_indices(d)])
        neighbor_logits = torch.zeros_like(logits).log()
        neighbor_logits[valid_next_step_indices] = logits.index_select(0, valid_next_step_indices)
        return neighbor_logits
