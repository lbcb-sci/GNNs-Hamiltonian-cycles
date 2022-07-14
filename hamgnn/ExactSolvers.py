import os
import re
import shutil
import subprocess
import textwrap
from typing import List
import copy
import time
import threading
import warnings

import torch
import torch_geometric as torch_g

from hamgnn.HamiltonSolver import HamiltonSolver
from hamgnn.constants import CONCORDE_SCRIPT_PATH, CONCORDE_WORK_DIR, CONCORDE_INPUT_FILE


class ConcordeHamiltonSolver(HamiltonSolver):
    def __init__(self, root_dir=CONCORDE_WORK_DIR, working_subdir=None):
        self.CONCORDE_EXECUTABLE_PATH = CONCORDE_SCRIPT_PATH
        if working_subdir is None:
            working_subdir = f"thread-{threading.get_ident()}_time-{time.time_ns()}"
        self.working_dir = os.path.join(root_dir, working_subdir)
        self.CONCORDE_INPUT_FILE = CONCORDE_INPUT_FILE

    def init_work_folder(self):
        if not os.path.isdir(self.working_dir):
            os.makedirs(self.working_dir, exist_ok=True)

    def clean(self):
        if os.path.isdir(self.working_dir):
            shutil.rmtree(self.working_dir)

    def __del__(self):
        self.clean()

    def create_input_file(self, d: torch_g.data.Data, filepath=None, name="Unnamed_instance.tsp"):
        if filepath is None:
            filepath = os.path.join(self.working_dir, self.CONCORDE_INPUT_FILE)

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
        with open(filepath, "w") as out:
            out.write(out_string)

    # INPUT FILE CREATION TAKES A LONG TIME (QUADRATIC COMPLEXITY) SO THE RESULTS CAN BE UNINTUITIVE!
    def time_execution(self, d: torch_g.data.Data, input_file=None, is_consume_input_file=False):
        # TODO NEED TO CHECK IF CONCORDE IS PROPERLY INSTALLED!
        if input_file is None:
            self.create_input_file(d)
            is_consume_input_file=True
            input_file = os.path.join(self.working_dir, self.CONCORDE_INPUT_FILE)
        input_file = os.path.abspath(input_file)
        called_process = subprocess.run(["time", self.CONCORDE_EXECUTABLE_PATH, input_file],
                                        cwd=self.working_dir, capture_output=True)
        string_stdout, string_stderr = [
            out_stream.decode("utf-8") for out_stream in [called_process.stdout, called_process.stderr]]

        optimal_len_regex = re.search(r"Optimal Solution: ([0-9.]+)", string_stdout)
        if optimal_len_regex is None:
            warnings.warn("Concorde printout tends to have problem with certain small graphs like a triangle graph. Checked it by hand and it really"
                          " seems to be a bug. Ignoring and reading the solution as if everything were normal.")
            optimal_len = d.num_nodes
        else:
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

        if len(tour) > 0:
            # Empty list indicates no solution. Otherwise it is a cycle but the end point is not the starting point
            tour.append(tour[0])

        if is_consume_input_file:
            os.remove(input_file)

        return tour, real_time, user_time

    @staticmethod
    def _transform_path_problem_graph_to_cycles_problem_path(d: torch_g.data.Data):
        transformed_d = copy.deepcopy(d)
        artifiical_node = d.num_nodes
        transformed_d.num_nodes += 1

        all_nodes = torch.arange(0, d.num_nodes, dtype=d.edge_index.dtype, device=d.edge_index.device)
        new_edges = torch.stack([all_nodes, torch.full_like(all_nodes, artifiical_node)])
        transformed_d.edge_index = torch.concat([transformed_d.edge_index, new_edges, new_edges.flip(0)], dim=-1)

        return transformed_d, artifiical_node


    def solve(self, d: torch_g.data.Data, is_only_look_for_cylce=False):
        hamiltonian_cycle, _, _ = self.time_execution(d)
        if is_only_look_for_cylce or len(hamiltonian_cycle) > 0:
            return hamiltonian_cycle

        ham_path_d, artificial_node = self._transform_path_problem_graph_to_cycles_problem_path(d)
        artificial_cycle, _, _ = self.time_execution(ham_path_d)
        if len(artificial_cycle) > 0:
            artificial_cycle = artificial_cycle[:-1]
            artificial_node_index = artificial_cycle.index(artificial_node)
            hamiltonian_path = artificial_cycle[artificial_node_index + 1:] + artificial_cycle[:artificial_node_index]
            return hamiltonian_path

        return []

    def solve_graphs(self, graphs: List[torch_g.data.Data]):
        return [self.solve(graph) for graph in graphs]

    def timeed_solve_graphs(self, graphs: List[torch_g.data.Data]):
        return zip([self.time_execution(g) for g in graphs])