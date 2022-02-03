import os
import re
import shutil
import subprocess
import textwrap
from typing import List

import torch
import torch_geometric as torch_g
from src.HamiltonSolver import HamiltonSolver

from src.constants import CONCORDE_SCRIPT_PATH, CONCORDE_WORK_DIR, CONCORDE_INPUT_FILE

class ConcordeHamiltonSolver(HamiltonSolver):
    def __init__(self, root_dir=CONCORDE_WORK_DIR, working_subdir=""):
        self.CONCORDE_EXECUTABLE_PATH = CONCORDE_SCRIPT_PATH
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

    def create_input_file(self, d: torch_g.data.Data, filename=None, name="Unnamed_instance.tsp"):
        if filename is None:
            filename = os.path.join(self.working_dir, self.CONCORDE_INPUT_FILE)

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
        with open(filename, "w") as out:
            out.write(out_string)

    # INPUT FILE CREATION TAKES A LONG TIME (QUADRATIC COMPLEXITY) SO THE RESULTS CAN BE UNINTUITIVE!
    def time_execution(self, d: torch_g.data.Data, input_file=None, is_consume_input_file=False):
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

    def solve(self, d: torch_g.data.Data, input_file=None):
        return self.time_execution(d, input_file)[0]

    def solve_graphs(self, graphs: List[torch_g.data.Data]):
        return [self.solve(graph) for graph in graphs]