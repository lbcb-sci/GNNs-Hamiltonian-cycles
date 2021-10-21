import itertools
import networkx as nx
import os
import re
import shutil
import subprocess
import textwrap

import torch
import torch_geometric as torch_g

from src.constants import CONCORDE_SCRIPT_PATH, CONCORDE_WORK_DIR, CONCORDE_INPUT_FILE


def is_hamilton_cycle(G: nx.DiGraph, edges):
    visited_nodes = set()
    current = next(iter(G.nodes))
    while current not in visited_nodes:
        visited_nodes.add(current)
        out = [e for (s, e) in edges if s == current]
        if len(out) != 1:
            return False
        current = out[0]
    return len(visited_nodes) == len(G.nodes)


def dynam_prog_find_hamiltonian_cycles(G: nx.DiGraph):
    known_paths = {}
    if len(G.nodes) == 0:
        raise Exception("Looking for Hamilton cycle in empty graph")
    origin = next(iter(G.nodes))

    for n in G.succ[origin]:
        known_paths[(n,)] = [(origin, n)]

    stack = [tuple(sorted(x for x in G.nodes if x != origin)) + (origin,)]

    while stack:
        plan = stack[0]
        stack = stack[1:]
        unique_id = object()
        paths = known_paths.get(plan, unique_id)
        if paths != unique_id:
            continue
        simplifications = [tuple(sorted(x for x in plan[:-1] if x != n)) + (n, ) for n in G.pred[plan[-1]] if n in plan]
        if len(simplifications) == 0:
            known_paths[plan] = []
            continue
        not_computed = []
        precomputed = []
        for s in simplifications:
            v = known_paths.get(s, unique_id)
            if v == unique_id:
                not_computed += [s]
            else:
                precomputed += v
        if len(not_computed) == 0:
            known_paths[plan] = [pre + (plan[-1],) for pre in precomputed]
        else:
            stack = not_computed + [plan] + stack
    return known_paths[tuple(sorted(x for x in G.nodes if x != origin)) + (origin,)]


class ConcordeHamiltonSolver:
    def __init__(self):
        self.CONCORDE_SCRIPT_PATH = CONCORDE_SCRIPT_PATH
        self.CONCORDE_WORK_DIR = CONCORDE_WORK_DIR
        self.CONCORDE_INPUT_FILE = CONCORDE_INPUT_FILE

    def init_work_folder(self):
        if not os.path.isdir(self.CONCORDE_WORK_DIR):
            os.mkdir(self.CONCORDE_WORK_DIR)

    def clean(self):
        if os.path.isdir(self.CONCORDE_WORK_DIR):
            shutil.rmtree(self.CONCORDE_WORK_DIR)

    def __del__(self):
        self.clean()

    def create_input_file(self, d: torch_g.data.Data, output_file=None, name="Unnamed_instance.tsp"):
        if output_file is None:
            output_file = os.path.join(self.CONCORDE_WORK_DIR, self.CONCORDE_INPUT_FILE)

        adjacency = torch.sparse_coo_tensor(
            d.edge_index, torch.ones(d.edge_index.shape[1], dtype=torch.int, device=d.edge_index.device),
            (d.num_nodes, d.num_nodes))
        adjacency = adjacency.to_dense()
        weights = 2*torch.ones_like(adjacency) - adjacency
        weights_str = ''.join([''.join([str(x.item()) + " " for x in weights[i]] + ['\n']) for i in range(d.num_nodes)])
        self.init_work_folder()
        out_string = textwrap.dedent(
            """
            NAME: {}
            TYPE: TSP
            COMMENT: None
            DIMENSION: {}
            EDGE_WEIGHT_TYPE: EXPLICIT
            EDGE_WEIGHT_FORMAT: FULL_MATRIX
            EDGE_WEIGHT_SECTION
            {}""").format(name, d.num_nodes, weights_str)
        with open(output_file, "w") as out:
            out.write(out_string)

    def time_execution(self, d: torch_g.data.Data, input_file=None):
        if input_file is None:
            self.create_input_file(d)
            input_file = os.path.join(self.CONCORDE_WORK_DIR, self.CONCORDE_INPUT_FILE)
        input_file = os.path.abspath(input_file)
        called_process = subprocess.run(["time", self.CONCORDE_SCRIPT_PATH, input_file],
                                        cwd=self.CONCORDE_WORK_DIR, capture_output=True)
        string_stdout, string_stderr = [
            out_stream.decode("utf-8") for out_stream in [called_process.stdout, called_process.stderr]]

        optimal_len_regex = re.search(r"Optimal Solution: ([0-9.]+)", string_stdout)
        optimal_len = round(float(optimal_len_regex.group(0).split(":")[1].strip()))

        user_time_regex = re.search(r"([0-9.]+)user", string_stderr)
        real_time_regex = re.search(r"([0-9.]+):([0-9.]+)elapsed", string_stderr)
        user_time = float(user_time_regex.group(1))
        real_time = float(real_time_regex.group(1)) * 60 + float(real_time_regex.group(2))

        if optimal_len != d.num_nodes:
            tour = []
        else:
            base_name, extension = os.path.splitext(input_file)
            with open(base_name + ".sol", "r") as result:
                lines = result.readlines()
                tour = [int(x) for i in range(1, len(lines)) for x in lines[i].split()]

        return tour, real_time, user_time

    def solve(self, d: torch_g.data.Data, input_file=None):
        return self.time_execution(d, input_file)[0]


if __name__ == '__main__':
    num_nodes = 20
    edge_existence_prob = 0.8
    nr_examples = 10
    print(f"Testing Concorde TSP on {nr_examples} random Erdos-Reny graphs of size {num_nodes}"
          f" with edge existence probability {edge_existence_prob}.")
    from src.GraphGenerators import ErdosRenyiGenerator

    generator = ErdosRenyiGenerator(num_nodes, edge_existence_prob)
    for d in itertools.islice(generator, nr_examples):
        concorde = ConcordeHamiltonSolver()
        tour = concorde.solve(d)
        print(tour)
