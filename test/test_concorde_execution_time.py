from tqdm import tqdm
import platform
import subprocess
import itertools
import pandas

from src.GraphGenerators import ErdosRenyiGenerator
from src.ExactSolvers import ConcordeHamiltonSolver

if __name__ == "__main__":
    graph_sizes = [2**k for k in range(5, 15)]
    examples_per_size = 3
    ham_existence_prob = 0.8
    print(f"Checking concorde execution time on Erdos-Renyi graphs with {ham_existence_prob} asysmptotic probability of being Hamiltonian."
          f"Timing execution {examples_per_size} times per each graph_size in {graph_sizes}")
    if platform.system() == "Linux":
        processor_model_name = subprocess.check_output('lscpu | grep "Model name"', shell=True, text=True).split(":")[-1].strip()
    else:
        processor_model_name = "unknown"
    print(f"Processor model: {processor_model_name} (Concorde is a single-thread program)")
    concorde_solver = ConcordeHamiltonSolver()
    
    _sizes_column = []
    _times_column = []
    for size in tqdm(graph_sizes):
        generator = ErdosRenyiGenerator(size, ham_existence_prob)
        for graph in tqdm(itertools.islice(generator, examples_per_size), leave=False, total=examples_per_size):
            exec_time = concorde_solver.time_execution(graph)[1]
            _sizes_column.append(size)
            _times_column.append(exec_time)
    df_times = pandas.DataFrame({"graph_size": _sizes_column, "execution_time": _times_column})
    df_times.to_csv("concorde_execution_times.csv")
    print(df_times.groupby("graph_size").agg("mean"))