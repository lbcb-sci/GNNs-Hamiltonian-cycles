from tqdm import tqdm
import platform
import subprocess
import itertools
import pandas
import time
from pathlib import Path

from src.data.GraphGenerators import ErdosRenyiGenerator
from src.ExactSolvers import ConcordeHamiltonSolver


def test_concorde_execution_time():
    graph_sizes = [int((1.5)**k) for k in range(10, 24)]
    ham_existance_probabilites = [0.1, 0.3, 0.5, 0.7]
    examples_per_size = 5
    timestamp = int(time.time())

    print(f"Checking concorde execution time on Erdos-Renyi graphs with {ham_existance_probabilites} asysmptotic probability of being Hamiltonian."
          f"Timing execution {examples_per_size} times per each graph_size in {graph_sizes}")
    if platform.system() == "Linux":
        processor_model_name = subprocess.check_output('lscpu | grep "Model name"', shell=True, text=True).split(":")[-1].strip()
    else:
        processor_model_name = "unknown"
    print(f"Processor model: {processor_model_name} (Concorde is a single-thread program)")
    concorde_solver = ConcordeHamiltonSolver(working_subdir="test_execution_time")

    _sizes_column = []
    _real_time_column = []
    _user_time_column = []
    _is_hamiltonian_column = []
    _ham_existence_prob_column = []
    for size, prob in tqdm(list(itertools.product(graph_sizes, ham_existance_probabilites))):
        generator = ErdosRenyiGenerator(size, prob)
        for graph in tqdm(itertools.islice(generator, examples_per_size), leave=False, total=examples_per_size):
            solution, real_time, user_time = concorde_solver.time_execution(graph)
            _sizes_column.append(size)
            _real_time_column.append(real_time)
            _user_time_column.append(user_time)
            _is_hamiltonian_column.append(len(solution) == (size + 1))
            _ham_existence_prob_column.append(prob)

    df_times = pandas.DataFrame({"graph_size": _sizes_column, "real_execution_time": _real_time_column,
                                 "user_execution_time": _user_time_column, "is_hamiltonian": _is_hamiltonian_column,
                                 "timestamp": timestamp, "HC_existence_probability": _ham_existence_prob_column})
    return df_times


if __name__ == "__main__":
    df_path = Path(__file__).parent / "concorde_execution_times.csv"
    df_new = test_concorde_execution_time()
    df_old = pandas.read_csv(df_path)
    df = pandas.concat([df_old, df_new])
    df.to_csv(df_path, index=False)
    print(df.groupby("graph_size").agg("mean"))