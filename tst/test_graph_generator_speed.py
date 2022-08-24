from tqdm import tqdm
import pandas
import time
from pathlib import Path

from src.data.GraphGenerators import ErdosRenyiGenerator

def test_graph_generator_speed():
    nr_examples_per_size = 10
    sizes = [25, 50, 100, 500, 2000, 10_000, 50_000, 100_000]
    ham_existence_prob = 0.8
    _generation_time_column = []
    _size_column = []
    for size in tqdm(sizes):
        generator = iter(ErdosRenyiGenerator(size, ham_existence_prob))
        for _ in tqdm(range(nr_examples_per_size), total=nr_examples_per_size, leave=False):
            start_time_ns = time.process_time_ns()
            _ = next(generator)
            end_time_ns = time.process_time_ns()
            execution_time = (end_time_ns - start_time_ns) / 1000_000_000
            _size_column.append(size)
            _generation_time_column.append(execution_time)
    return pandas.DataFrame({"graph_size": _size_column, "time_to_generate": _generation_time_column})


if __name__ == "__main__":
    df = test_graph_generator_speed()
    df.to_csv("tests/graph_generation_time.csv")
    print(df.groupby("graph_size").agg("mean"))
