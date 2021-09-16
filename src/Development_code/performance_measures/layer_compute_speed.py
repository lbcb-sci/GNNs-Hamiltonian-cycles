import time
import itertools

import torch
import torch_geometric as torch_g
import torch_scatter

from src.DatasetBuilder import ErdosRenyiInMemoryDataset, ErdosRenyiGenerator
from src.Models import GatedGCNEmbedAndProcess, HamiltonianCycleFinder, EncodeProcessDecodeAlgorithm
from src.ExactSolvers import ConcordeHamiltonSolver


def time_operation(operation, dataset):
    total_time = 0
    nr_examples = 0
    result = None
    for d in dataset:
        start_time = time.time()
        result = operation(d)
        end_time = time.time()
        total_time += end_time - start_time
        nr_examples += 1
    return total_time / nr_examples, result


def time_operation_on_ER(operation, sizes=None, ham_prob=0.8, nr_examples_per_size=10):
    if sizes is None:
        sizes = [25, 100, 400, 1600, 5000]
    timings = []
    for s in sizes:
        generator = ErdosRenyiGenerator(s, ham_prob)
        t, choice = time_operation(operation, itertools.islice(generator, nr_examples_per_size))
        timings.append(t)
    return timings


def operation_forward_step(model: HamiltonianCycleFinder, d: torch_g.data.Data):
    model.init_graph(d)
    model.prepare_for_first_step(d, torch.tensor([0], device=d.edge_index.device))
    return model.next_step_prob_masked_over_neighbors(d)


def profile_EPD(model: EncodeProcessDecodeAlgorithm, d):
    times = {}
    d = torch_g.data.Batch.from_data_list([d])
    times["data_init"], _ = time_operation(lambda a: model.init_graph(a), [d])
    times["data_prep"], _ = time_operation(lambda a: model.prepare_for_first_step(a, [0]), [d])
    times["pure_logits"], l = time_operation(lambda a: model.next_step_logits(d), [d])
    times["neighbor_logits"], n = time_operation(lambda a: model._mask_neighbor_logits(l, a), [d])
    times["probs_computation"], p = time_operation(lambda a: torch_scatter.scatter_softmax(n, a.batch), [d])

    def greedy_choice(p):
        choice = torch.argmax(
            torch.isclose(p, torch.max(p, dim=-1)[0][..., None]) * (p + torch.randperm(p.shape[-1])[None, ...]), dim=-1)
        choice = choice + torch_scatter.scatter_sum(d.batch, d.batch, dim_size=d.num_graphs)
        return choice

    times["selection"], choice = time_operation(lambda d: greedy_choice(p), [d])

    return times


if __name__ == '__main__':
    device_name = "cpu"

    HamS_model = EncodeProcessDecodeAlgorithm(True, processor_depth=5, hidden_dim=32, device=device_name)
    s1 = 500

    generator = ErdosRenyiGenerator(s1, 0.8)
    d = next(iter(generator)).to(device_name)
    times = profile_EPD(HamS_model, d)
    print(f"Operation times in s: {times}")
    total = sum(times.values())
    perc = {k: v/total for k, v in times.items()}
    print(f"Fraction of time spent on operation: {perc}")

    d = next(iter(generator)).to(device_name)
    d = torch_g.data.Batch.from_data_list([d])
    full, _ = time_operation(lambda a: operation_forward_step(HamS_model, a), [d])
    print(f"Single run, {full}, components total {total}")

    sizes = [s1]
    try:
        concorde_solver = ConcordeHamiltonSolver()
        timings = time_operation_on_ER(lambda d: concorde_solver.solve(d), sizes, nr_examples_per_size=1)
        print(f"Concorde solver: {timings}")
    except:
        print("Concorde not installed on the system")
    print("Repeated generation:", s1*full)
    timings = time_operation_on_ER(
        lambda d: HamS_model.batch_run_greedy_neighbor(torch_g.data.Batch.from_data_list([d])),
        sizes=sizes, nr_examples_per_size=1)
    print(f"HamS model greedy: {timings}")