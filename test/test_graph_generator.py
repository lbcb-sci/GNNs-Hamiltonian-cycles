from tqdm import tqdm
import itertools
import pandas

from src.GraphGenerators import ErdosRenyiGenerator
from src.ExactSolvers import ConcordeHamiltonSolver


if __name__ == "__main__":
    nr_examples = 100
    graph_sizes = [20, 40, 60, 80, 100, 125, 150, 175, 200, 250, 300, 400]
    ham_existence_probabilty = 0.8
    nr_hamiltonian_per_size = []
    concorde_solver = ConcordeHamiltonSolver()
    print(f"Checking the fraction of Hamiltonian graphs for sizes {graph_sizes} (should be asymptotically {ham_existence_probabilty})")
    for graph_size in tqdm(graph_sizes):
        generator = ErdosRenyiGenerator(graph_size, ham_existence_probabilty)
        nr_hamiltonian = 0
        for graph in tqdm(list(itertools.islice(generator, nr_examples)), leave=False):
            solution = concorde_solver.solve(graph)
            if len(solution) == graph_size + 1:
                nr_hamiltonian += 1
            nr_hamiltonian_per_size.append(nr_hamiltonian)
    df_hamiltonian = pandas.DataFrame({"graph_size": graph_sizes, "nr_Hamiltonian": nr_hamiltonian_per_size})
    df_hamiltonian["nr_graphs_generated"] = nr_examples
    df_hamiltonian["Hamiltonian_fraction"] = df_hamiltonian["nr_Hamiltonian"] / df_hamiltonian["nr_graphs_generated"]
    df_hamiltonian.to_csv("fraction_hamiltonian_graphs.csv")
    print(f"Fractions are as follows: {df_hamiltonian['fraction_hamiltonian_graphs']}")
    