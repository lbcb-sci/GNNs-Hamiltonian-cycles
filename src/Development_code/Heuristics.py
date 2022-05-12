from pathlib import Path
import networkit
import networkx as nx
import torch
from copy import deepcopy

from src.Models import HamiltonSolver

def _to_networkit(num_nodes, edge_index: torch.tensor):
    g = networkit.Graph(num_nodes)
    for edge in edge_index.t():
        if not g.hasEdge(*edge):
            g.addEdge(*edge)
    return g


def _least_degree_first(g: networkit.Graph, start_node, is_use_unreachable_vertex_heuristics=True):
    g = deepcopy(g)
    current = start_node
    path = []
    for step in range(g.numberOfNodes()):
        path.append(current)
        next_step_options = sorted([x for x in g.iterNeighbors(current)], key=g.degree)
        g.removeNode(current)
        if is_use_unreachable_vertex_heuristics:
            better_options = [x for x in next_step_options
                              if g.degree(x) > 0 and min([g.degree(y) for y in g.iterNeighbors(x)]) > 1]
            if better_options:
                next_step_options = better_options

        if next_step_options:
            current = next_step_options[0]
        else:
            break
    return path


def least_degree_first_heuristics(num_nodes, edge_index: torch.tensor, is_use_unreachable_vertex_heuristics=True):
    g = _to_networkit(num_nodes, edge_index)
    max_degree = networkit.graphtools.maxDegree(g)
    start_node = [x for x in g.iterNodes() if g.degree(x) == max_degree][0]
    path = _least_degree_first(g, start_node, is_use_unreachable_vertex_heuristics)
    if len(path) == num_nodes and g.hasEdge(path[0], path[-1]):
        path.append(start_node)
    return path


def _rotational_options(g, path):
    end = path[-1]
    rotational_options = []
    for i in range(len(path) - 2):
        if g.hasEdge(path[i], end) and [y for y in g.iterNeighbors(path[i + 1]) if y not in path]:
            rotational_options.append(path[:i + 1] + [path[j] for j in range(len(path) - 1, i, -1)])
    return rotational_options


def _invert_path(path):
    return [path[i] for i in range(len(path) - 1, -1, -1)]

class LeastDegreeFirstHeuristics(HamiltonSolver):
    def solve_graphs(self, graphs):
        return [least_degree_first_heuristics(graph.num_nodes, graph.edge_index, True) for graph in graphs]

class HybridHam(HamiltonSolver):
    def solve_graphs(self, graphs):
        return [self._solve(graph.num_nodes, graph.edge_index) for graph in graphs]

    def _solve(self, num_nodes, edge_index: torch.tensor):
        g = _to_networkit(num_nodes, edge_index)
        max_degree = networkit.graphtools.maxDegree(g)
        start_options = [x for x in g.iterNodes() if g.degree(x) == max_degree]
        initial_paths = []
        for start_node in start_options:
            initial_paths.append(_least_degree_first(g, start_node, True))
        path = max(initial_paths, key=len)

        if len(path) <= 2:
            return path

        while len(path) < num_nodes:
            if g.degree(path[0]) > g.degree(path[-1]):
                path = _invert_path(path)
            rotational_options = _rotational_options(g, path)
            if rotational_options:
                reduced_graph = deepcopy(g)
                for edge in g.iterEdges():
                    if edge[0] in path or edge[1] in path:
                        reduced_graph.removeEdge(*edge)
                extension_options = [p for p in rotational_options if reduced_graph.degree(p[-1]) > 0]
                if not extension_options:
                    return path
                path = max(extension_options, key=lambda p: reduced_graph.degree(p[-1]))
                path = path + _least_degree_first(reduced_graph, path[-1])
            else:
                return path

        if g.hasEdge(path[0], path[-1]):
            path.append(path[0])
            return path

        if g.degree(path[0]) < g.degree(path[-1]):
            path = _invert_path(path)

        rotational_options = _rotational_options(g, path)
        for option in rotational_options:
            if g.hasEdge(option[0], option[-1]):
                option = option.append(option[0])
                return option
        return path


def load_graph_from_hcp_file(hcp_path):
    lines = Path(hcp_path).read_text().split("\n")
    _dim_line = next(l for l in lines if l.startswith("DIMENSION"))
    num_nodes = int(_dim_line.split()[-1])
    edges_start, edges_end = None, None
    for i in range(len(lines)):
        if lines[i].startswith("EDGE_DATA_SECTION"):
            edges_start = i + 1
        if lines[i].startswith("-1"):
            edges_end = i
    edges_lines = lines[edges_start: edges_end]
    edge_index = torch.tensor([[int(node) for node in l.split()] for l in edges_lines]).t()
    # .hcp start indexing nodes for 1 insetad of 0
    edge_index -= 1
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
    return num_nodes, edge_index


if __name__ == '__main__':
    from src.data.GraphGenerators import ErdosRenyiGenerator
    import itertools
    import torch_geometric
    import networkx
    from matplotlib import pyplot as plt
    from copy import deepcopy

    from pathlib import Path
    path = Path(__file__).parent.parent.parent / "HCP_benchmarks/graph2.hcp"
    num_nodes, edge_index = load_graph_from_hcp_file(path)
    graph = torch_geometric.data.Data(num_nodes=num_nodes, edge_index=edge_index)
    solution = least_degree_first_heuristics(num_nodes, edge_index)
    print(solution)


    num_nodes = 100
    generator = ErdosRenyiGenerator(num_nodes, 0.4)

    for d in itertools.islice(generator, 100):
        path = least_degree_first_heuristics(num_nodes, d.edge_index, False)
        path_improved = least_degree_first_heuristics(num_nodes, d.edge_index, True)
        HybridHam_path = HybridHam().solve(d)
        print(len(HybridHam_path), len(path_improved))
        if len(HybridHam_path) == num_nodes + 1:
            nx_graph = torch_geometric.utils.to_networkx(d)
            pos = networkx.spring_layout(nx_graph)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            for ax in (ax1, ax2):
                nx.draw(nx_graph, pos, with_labels=True, ax=ax, arrows=False)

            HybridHam_path_edges = [(HybridHam_path[i], HybridHam_path[i + 1]) for i in range(len(HybridHam_path) - 1)]
            nx_graph.remove_edges_from(deepcopy(nx_graph.edges))
            nx_graph.add_edges_from(HybridHam_path_edges)
            nx.draw_networkx_edges(nx_graph, pos, edge_color="red", arrows=False, ax=ax2)
            plt.show()
