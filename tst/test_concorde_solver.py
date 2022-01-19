import itertools
from src.ExactSolvers import ConcordeHamiltonSolver
from src.GraphGenerators import ErdosRenyiGenerator

if __name__ == '__main__':
    num_nodes = 50
    hamiltonian_cylce_existence_prob = 0.8
    
    nr_examples = 10
    print(f"Testing Concorde TSP on {nr_examples} random Erdos-Reny graphs of size {num_nodes}"
          f" with Hamiltonian cycle existence probabilty {hamiltonian_cylce_existence_prob}.")

    generator = ErdosRenyiGenerator(num_nodes, hamiltonian_cylce_existence_prob)
    concorde_solver = ConcordeHamiltonSolver()
    for d in itertools.islice(generator, nr_examples):
        tour = concorde_solver.solve(d)
        print(tour)
        
    to_generate = 100
    print(f"Checking the number of Hamiltonian graphs out of {to_generate}")
    graphs = list(itertools.islice(generator, to_generate))
    solutions = concorde_solver.solve_graphs(graphs)
    print(f"{len([path for path in solutions if len(path) == num_nodes + 1])} out"
          f"of {to_generate} graphs are Hamiltonian")