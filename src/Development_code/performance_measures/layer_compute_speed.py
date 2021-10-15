import time
import itertools

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch_geometric as torch_g
import torch_scatter

from src.DatasetBuilder import ErdosRenyiInMemoryDataset, ErdosRenyiGenerator
from src.Models import GatedGCNEmbedAndProcess, HamiltonianCycleFinder, EncodeProcessDecodeAlgorithm
from src.ExactSolvers import ConcordeHamiltonSolver
from src.NN_modules import ResidualMultilayerMPNN


def time_operation(operation, dataset):
    total_time = 0
    nr_examples = 0
    result = None
    for d in dataset:
        start_time = time.time_ns()
        result = operation(d)
        end_time = time.time_ns()
        total_time += (end_time - start_time) // 1e9
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


def profile_EPD(model: EncodeProcessDecodeAlgorithm, d: torch_g.data.Batch):
    times = {}
    times["data_init"], _ = time_operation(lambda a: model.init_graph(a), [d])
    times["data_prep"], _ = time_operation(lambda a: model.prepare_for_first_step(a, [0]), [d])

    times["encoding"], d.z = time_operation(lambda a: model.encoder_nn(torch.cat([a.x, a.h], dim=-1)), [d])
    times["processing"], d.h = time_operation(lambda a: model.processor_nn(a.z, a.edge_index, a.edge_attr), [d])
    times["decoding"], l = time_operation(lambda a: torch.squeeze(model.decoder_nn(torch.cat([a.z, a.h], dim=-1)), dim=-1), [d])
    # times["pure_logits"], l = time_operation(lambda a: model.next_step_logits(a), [d])
    times["neighbor_logits"], n = time_operation(lambda a: model._mask_neighbor_logits(l, a), [d])
    times["probs_computation"], p = time_operation(lambda a: torch_scatter.scatter_softmax(n, a.batch), [d])

    def greedy_choice(p):
        choice = torch.argmax(
            torch.isclose(p, torch.max(p, dim=-1)[0][..., None])
            * (p + torch.randperm(p.shape[-1], device=p.device)[None, ...]), dim=-1)
        choice = choice + torch_scatter.scatter_sum(d.batch, d.batch, dim_size=d.num_graphs)
        return choice

    times["selection"], choice = time_operation(lambda d: greedy_choice(p), [d])

    return times


def profile_ResidualMPNN(net: ResidualMultilayerMPNN, x, edge_index, edge_weight):
    timings = {}
    layer_index =0
    for l in net.layers:
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("inference"):
                l.forward(x, edge_index, edge_weight)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        timings[f"layer{layer_index}x1"], x = time_operation(lambda a: l.forward(x, edge_index, edge_weight), [x])

        layer_index += 1
    print(timings)


if __name__ == '__main__':
    device_name = "cuda"
    s1 = 10_000
    number_format = "%.6f"
    d = next(iter(ErdosRenyiGenerator(s1, 0.8))).to(device_name)

    HamS_model = EncodeProcessDecodeAlgorithm(False, processor_depth=2, hidden_dim=32, device=device_name)
    net = HamS_model.processor_nn
    with torch.no_grad():
        HamS_model.init_graph(d)
        HamS_model.prepare_for_first_step(d, 0)
        x = HamS_model.encoder_nn.forward(torch.cat([d.x, d.h], dim=-1))
        edge_index, edge_weight = d.edge_index, d.edge_attr
        profile_ResidualMPNN(net, x, edge_index, edge_weight)

    generator = ErdosRenyiGenerator(s1, 0.8)
    d = torch_g.data.Batch.from_data_list([next(iter(generator))]).to(device_name)
    times = profile_EPD(HamS_model, d)
    formatted_times = {k: number_format % t for k, t in times.items()}
    print(f"Operation times in s: {formatted_times}")
    total = sum(times.values())
    formatted_perc = {k: number_format % (v/(total + 1e-8)) for k, v in times.items()}
    print(f"Fraction of time spent on operation: {formatted_perc}")

    d = next(iter(generator)).to(device_name)
    d = torch_g.data.Batch.from_data_list([d]).to(device_name)
    full, _ = time_operation(lambda a: operation_forward_step(HamS_model, a), [d])
    print(f"Forward step, {number_format % full}, components total {number_format % total}")

    sizes = [s1]
    try:
        concorde_solver = ConcordeHamiltonSolver()
        timings = time_operation_on_ER(lambda d: concorde_solver.solve(d), sizes, nr_examples_per_size=1)
        formatted_timings = [number_format % t for t in timings]
        print(f"Concorde solver: {timings}")
    except:
        print("Concorde not installed on the system")
    print(f"Repeated generation: {number_format % (s1*full)}")

    # Careful, this makes sense only for trained models. Otherwise greedy search terminates after just a few steps
    timings, _ = time_operation(
        lambda a: HamS_model.batch_run_greedy_neighbor(a), [d])
    print(f"HamS model greedy: {timings}")

    for i in range(3):
        timings, _ = time_operation(lambda a: HamS_model.next_step_prob_masked_over_neighbors(a), [d])
        print(f"Rerun {i}: {timings}")
