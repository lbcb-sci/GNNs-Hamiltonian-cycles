import time
import itertools
from matplotlib import pyplot as plt
import seaborn
import pandas
import pathlib

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch_geometric as torch_g
import torch_scatter

from src.DatasetBuilder import ErdosRenyiInMemoryDataset, ErdosRenyiGenerator
from src.Models import GatedGCNEmbedAndProcess, HamiltonianCycleFinder, EncodeProcessDecodeAlgorithm
from src.ExactSolvers import ConcordeHamiltonSolver
from src.NN_modules import ResidualMultilayerMPNN


def _time_operation_cpu(operation, dataset):
    total_time = 0
    nr_examples = 0
    for d in dataset:
        start_time = time.time_ns()
        result = operation(d)
        end_time = time.time_ns()
        total_time += (end_time - start_time) / 1e9
        nr_examples += 1
    return total_time / nr_examples, result


def _time_operation_cuda(operation, dataset):
    nr_examples = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for d in dataset:
        result = operation(d)
        nr_examples += 1
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000 / nr_examples, result


def time_operation(operation, dataset, device):
    if device == "cpu":
        return _time_operation_cpu(operation, dataset)
    elif device == "cuda":
        return _time_operation_cuda(operation, dataset)


def time_operation_on_ER(device, operation, sizes=None, ham_prob=0.8, nr_examples_per_size=10):
    if sizes is None:
        sizes = [25, 100, 400, 1600, 5000]
    timings = []
    for s in sizes:
        generator = ErdosRenyiGenerator(s, ham_prob)
        t, choice = time_operation(operation, itertools.islice(generator, nr_examples_per_size), device)
        timings.append(t)
    return timings


def operation_forward_step(model: HamiltonianCycleFinder, d: torch_g.data.Batch):
    model.init_graph(d)
    model.prepare_for_first_step(d, torch.tensor([0], device=d.edge_index.device))
    return model.next_step_prob_masked_over_neighbors(d)


def profile_data_preparation(model: HamiltonianCycleFinder, d: torch_g.data.Batch):
    times = {}
    times["data_init"], _ = time_operation(lambda a: model.init_graph(a), [d], model.device)
    times["data_prep"], _ = time_operation(lambda a: model.prepare_for_first_step(a, [0]), [d], model.device)
    return times, d


def profile_EPD(model: EncodeProcessDecodeAlgorithm, d: torch_g.data.Batch):
    times, d = profile_data_preparation(model, d)

    times["encoding"], d.z = time_operation(
        lambda a: model.encoder_nn(torch.cat([a.x, a.h], dim=-1)), [d], model.device)
    times["processing"], d.h = time_operation(
        lambda a: model.processor_nn(a.z, a.edge_index, a.edge_attr), [d], model.device)
    times["decoding"], l = time_operation(
        lambda a: torch.squeeze(model.decoder_nn(torch.cat([a.z, a.h], dim=-1)), dim=-1), [d], model.device)
    times["neighbor_logits"], n = time_operation(
        lambda a: model._mask_neighbor_logits(l, a), [d], model.device)
    times["probs_computation"], p = time_operation(
        lambda a: torch_scatter.scatter_softmax(n, a.batch), [d], model.device)

    def greedy_choice(p):
        choice = torch.argmax(
            torch.isclose(p, torch.max(p, dim=-1)[0][..., None])
            * torch.randperm(p.shape[-1], device=p.device)[None, ...], dim=-1)
        choice = choice + torch_scatter.scatter_sum(d.batch, d.batch, dim_size=d.num_graphs)
        return choice

    times["selection"], choice = time_operation(lambda d: greedy_choice(p), [d], model.device)

    return times


def profile_ResidualMPNN(net: ResidualMultilayerMPNN, x, edge_index, edge_weight):
    for l in net.layers:
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("inference"):
                l.forward(x, edge_index, edge_weight)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))


if __name__ == '__main__':
    figure_output_path = pathlib.Path("src/Development_code/plots_and_test_results/")

    main_device = "cuda"
    number_format = "%.6f"

    local_devices = ["cpu"]
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 3:
        local_devices += ["cuda"]

    HamS_model = EncodeProcessDecodeAlgorithm(True, processor_depth=5, hidden_dim=32, device=main_device)
    HamS_processor_net = HamS_model.processor_nn

    forward_pass_sizes = list(range(100, 10_000, 100))
    records = []

    with torch.no_grad():
        for l_device in local_devices:
            # Device warmup
            _ = torch.einsum("ij,jk -> ik", *[torch.rand([5_000, 5_000], device=main_device) for _ in range(2)])
            HamS_model.to(l_device)
            for s in forward_pass_sizes:
                graph_generator = ErdosRenyiGenerator(s, 0.8)
                d = torch_g.data.Batch.from_data_list([next(iter(graph_generator))]).to(HamS_model.device)
                forward_pass_s, _ = time_operation(
                    lambda a: operation_forward_step(HamS_model, a), [d], HamS_model.device)

                HamS_model.init_graph(d)
                HamS_model.prepare_for_first_step(d, 0)
                x = HamS_model.encoder_nn.forward(torch.cat([d.x, d.h], dim=-1))
                edge_index = d.edge_index
                edge_weight = d.edge_attr
                processor_forward_s, _ = time_operation(
                    lambda a: HamS_model.processor_nn.forward(x, edge_index, edge_weight), [d], HamS_model.device)

                records.append({"device": l_device, "size": s, "forward_pass_s": forward_pass_s,
                                "runtime_estimate_s": forward_pass_s * s, "processor_nn_s": processor_forward_s})
    df_forward_pass = pandas.DataFrame.from_records(records)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    seaborn.lineplot(x="size", y="runtime_estimate_s", data=df_forward_pass, ax=ax1, hue="device")
    seaborn.lineplot(x="size", y="forward_pass_s", data=df_forward_pass, ax=ax2, hue="device")
    plt.show()
    fig.savefig(figure_output_path / "pytorch_devices_inference_time")

    df_largest_graphs = df_forward_pass[df_forward_pass["size"] == df_forward_pass["size"].max()]
    fastest_device = df_largest_graphs[
        df_largest_graphs["forward_pass_s"] == df_largest_graphs["forward_pass_s"].min()]["device"].item()

    sizes = [1000]
    with torch.no_grad():
        for s in sizes:
            graph_generator = ErdosRenyiGenerator(s, 0.8)
            d = next(iter(graph_generator))
            # This tends to allocate too much memory on GPU
            d = d.to(main_device)

            HamS_model.init_graph(d)
            HamS_model.prepare_for_first_step(d, 0)
            x = HamS_model.encoder_nn.forward(torch.cat([d.x, d.h], dim=-1))
            edge_index, edge_weight = d.edge_index, d.edge_attr
            profile_ResidualMPNN(HamS_processor_net, x, edge_index, edge_weight)

            graph_generator = ErdosRenyiGenerator(s, 0.8)
            d = torch_g.data.Batch.from_data_list([next(iter(graph_generator))]).to(main_device)
            times = profile_EPD(HamS_model, d)
            formatted_times = {k: number_format % t for k, t in times.items()}
            print(f"Operation times in s: {formatted_times}")
            total = sum(times.values())
            formatted_perc = {k: number_format % (v/(total + 1e-8)) for k, v in times.items()}
            print(f"Fraction of time spent on operation: {formatted_perc}")

            # Careful, this makes sense only for trained models. Otherwise greedy search terminates after just a few steps
            timing, path = time_operation(
                lambda a: HamS_model.batch_run_greedy_neighbor(a), [d], HamS_model.device)
            print(f"Experimental graph runtime: {number_format % timing}. Path length={len(path)}"
                  f" -> projected graph runtime{number_format % (timing * s / len(path))}.")

            try:
                concorde_solver = ConcordeHamiltonSolver()
                timings = time_operation_on_ER(lambda d: concorde_solver.solve(d), sizes, nr_examples_per_size=1)
                formatted_timings = [number_format % t for t in timings]
                print(f"Concorde solver time: {number_format % timings}")
            except:
                print("Concorde not installed on the system")
