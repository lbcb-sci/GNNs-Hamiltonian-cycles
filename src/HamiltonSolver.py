import torch_geometric as torch_g
from abc import ABC, abstractmethod


class HamiltonSolver(ABC):
    @abstractmethod
    def solve_graphs(self, graphs: list[torch_g.data.Data]) -> list[list[int]]:
        pass

    def solve(self, graph: torch_g.data.Data):
        return self.solve_graphs([graph])[0]
