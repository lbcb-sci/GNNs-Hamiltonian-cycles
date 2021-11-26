import time
import itertools

import networkx as nx
import torch_geometric.utils
from matplotlib import pyplot as plt
import seaborn
import pandas
import pathlib

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch_geometric as torch_g
import torch_scatter

from src.DatasetBuilder import ErdosRenyiInMemoryDataset, ErdosRenyiGenerator
from src.Models import GatedGCNEmbedAndProcess, HamFinderGNN, EncodeProcessDecodeAlgorithm
from src.ExactSolvers import ConcordeHamiltonSolver
from src.NN_modules import ResidualMultilayerMPNN
from src.Development_code.Heuristics import least_degree_first_heuristics, HybridHam

from src.Evaluation import EvaluationScores, EvaluationPlots


def _time_operation_cpu(operation, dataset):
    total_time = 0
    nr_examples = 0
    results = []
    for d in dataset:
        start_time = time.process_time_ns()
        result = operation(d)
        end_time = time.process_time_ns()
        results.append(result)
        total_time += (end_time - start_time) / 1e9
        nr_examples += 1
    return total_time / nr_examples, results


def _time_operation_cuda(operation, dataset):
    nr_examples = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    results = []
    for d in dataset:
        results.append(operation(d))
        nr_examples += 1
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000 / nr_examples, results


def time_operation(operation, dataset, device):
    if device == "cpu":
        return _time_operation_cpu(operation, dataset)
    elif device == "cuda":
        return _time_operation_cuda(operation, dataset)

#
# def time_operation_on_ER(device, operation, sizes=None, ham_prob=0.8, nr_examples_per_size=10):
#     if sizes is None:
#         sizes = [25, 100, 400, 1600, 5000]
#     timings = []
#     for s in sizes:
#         generator = ErdosRenyiGenerator(s, ham_prob)
#         t, choice = time_operation(operation, itertools.islice(generator, nr_examples_per_size), device)
#         timings.append(t)
#     return timings


def operation_forward_step(model: HamFinderGNN, d: torch_g.data.Batch):
    model.init_graph(d)
    model.prepare_for_first_step(d, torch.tensor([0], device=d.edge_index.device))
    return model.next_step_prob_masked_over_neighbors(d)


def profile_data_preparation(model: HamFinderGNN, d: torch_g.data.Batch):
    times = {}
    times["data_init"], _ = time_operation(lambda a: model.init_graph(a), [d], model.device)
    times["data_prep"], _ = time_operation(lambda a: model.prepare_for_first_step(a, [0]), [d], model.device)
    return times, d


def profile_EPD(model: EncodeProcessDecodeAlgorithm, d: torch_g.data.Batch):
    runtime_records = []
    times, d = profile_data_preparation(model, d)

    times["encoding"], [d.z] = time_operation(
        lambda a: model.encoder_nn(torch.cat([a.x, a.h], dim=-1)), [d], model.device)
    times["processing"], [d.h] = time_operation(
        lambda a: model.processor_nn(a.z, a.edge_index, a.edge_attr), [d], model.device)
    times["decoding"], [l] = time_operation(
        lambda a: torch.squeeze(model.decoder_nn(torch.cat([a.z, a.h], dim=-1)), dim=-1), [d], model.device)
    times["neighbor_logits"], [n] = time_operation(
        lambda a: model._mask_neighbor_logits(l, a), [d], model.device)
    times["probs_computation"], [p] = time_operation(
        lambda a: torch_scatter.scatter_softmax(n, a.batch), [d], model.device)

    def greedy_choice(p):
        choice = torch.argmax(
            torch.isclose(p, torch.max(p, dim=-1)[0][..., None])
            * torch.randperm(p.shape[-1], device=p.device)[None, ...], dim=-1)
        choice = choice + torch_scatter.scatter_sum(d.batch, d.batch, dim_size=d.num_graphs)
        return choice

    times["selection"], _ = time_operation(lambda d: greedy_choice(p), [d], model.device)

    for k, v in times.items():
        runtime_records += [{"component": k, "runtime": v}]

    return runtime_records


def profile_ResidualMPNN(net: ResidualMultilayerMPNN, x, edge_index, edge_weight):
    for l in net.layers:
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("inference"):
                l.forward(x, edge_index, edge_weight)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))


def compare_forward_step_across_devices(model: EncodeProcessDecodeAlgorithm, sizes, hamilton_existance_prob):
    records = []

    for l_device in local_devices:
        # Device warmup
        _ = torch.einsum("ij,jk -> ik", *[torch.rand([5_000, 5_000], device=l_device) for _ in range(2)])
        model.to(l_device)
        for s in sizes:
            graph_generator = ErdosRenyiGenerator(s, hamilton_existance_prob)

            # This tends to allocate too much memory on GPU
            d = torch_g.data.Batch.from_data_list([next(iter(graph_generator))]).to(model.device)
            forward_pass_s, _ = time_operation(
                lambda a: operation_forward_step(model, a), [d], model.device)

            model.init_graph(d)
            model.prepare_for_first_step(d, 0)
            x = model.encoder_nn.forward(torch.cat([d.x, d.h], dim=-1))
            edge_index = d.edge_index
            edge_weight = d.edge_attr
            processor_forward_s, _ = time_operation(
                lambda a: model.processor_nn.forward(x, edge_index, edge_weight), [d], model.device)

            records.append({"device": l_device, "size": s, "forward_pass_s": forward_pass_s,
                            "runtime_estimate_s": forward_pass_s * s, "processor_nn_s": processor_forward_s})
        return pandas.DataFrame.from_records(records)


if __name__ == '__main__':
    with torch.no_grad():
        figure_output_path = pathlib.Path("src/Development_code/plots_and_test_results/")
        if not figure_output_path.exists():
            figure_output_path.mkdir()
        hamilton_existence_prob = 0.8
        profiler_row_limit = 20
        number_format = "%.6f"
        local_devices = ["cpu"]
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 3:
            local_devices += ["cuda"]

        HamS_model = EncodeProcessDecodeAlgorithm(True, processor_depth=5, hidden_dim=32, device=local_devices[0])
        HamS_processor_net = HamS_model.processor_nn

        # accuracy_comparison_sizes = [20, 50, 100, 150, 200]
        accuracy_comparison_sizes = [x for x in range(50, 400, 50)]
        accuracy_nr_examples_per_size = 20
        batch_size = 1
        HamS_evaluations = []
        least_degree_first_evaluation = []
        HybridHam_evaluation = []

        def _run_heuristic_on_batch_of_graphs(heuristic, batch: torch_g.data.Batch):
            node_per_graph = batch.num_nodes // batch.num_graphs
            tours = torch.full([batch.num_graphs, node_per_graph + 1], -1)
            data_list = batch.to_data_list()
            for data_index in range(len(data_list)):
                d = data_list[data_index]
                tour = heuristic(d.num_nodes, d.edge_index)
                for node_index in range(len(tour)):
                    tours[data_index, node_index] = tour[node_index]
            return tours, torch.empty([1])

        for s in accuracy_comparison_sizes:
            graph_generator = ErdosRenyiGenerator(s, hamilton_existence_prob)
            batch_generator = (
                torch_g.data.Batch.from_data_list([next(iter(graph_generator)) for data_index in range(batch_size)])
                for _ in itertools.count()
            )
            least_degree_first_evaluation.append(EvaluationScores.batch_evaluate(
                lambda batch: _run_heuristic_on_batch_of_graphs(least_degree_first_heuristics, batch),
                batch_generator, accuracy_nr_examples_per_size // batch_size
            ))

            HybridHam_evaluation.append(EvaluationScores.batch_evaluate(
                lambda batch: _run_heuristic_on_batch_of_graphs(HybridHam, batch),
                batch_generator, accuracy_nr_examples_per_size // batch_size
            ))

            # HamS_evaluations.append(EvaluationScores.batch_evaluate(
            #     lambda d: HamS_model.batch_run_greedy_neighbor(d),
            #     batch_generator, accuracy_nr_examples_per_size))

        # HamS_accuracy_scores = EvaluationScores.compute_accuracy_scores(HamS_evaluations, accuracy_comparison_sizes)

        for e in least_degree_first_evaluation + HybridHam_evaluation:
            e["nr_hamilton_graphs"] = 1
        EvaluationPlots.accuracy_curves(least_degree_first_evaluation, accuracy_comparison_sizes, best_expected_benchmark=0.8)
        plt.show()
        EvaluationPlots.accuracy_curves(HybridHam_evaluation, accuracy_comparison_sizes, best_expected_benchmark=0.8)
        plt.show()
        exit(-1)

        forward_pass_sizes = list(range(100, 1000, 100))
        df_forward_pass = compare_forward_step_across_devices(HamS_model, forward_pass_sizes, hamilton_existence_prob)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        seaborn.lineplot(x="size", y="runtime_estimate_s", data=df_forward_pass, ax=ax1, hue="device")
        seaborn.lineplot(x="size", y="forward_pass_s", data=df_forward_pass, ax=ax2, hue="device")
        fig.savefig(figure_output_path / "pytorch_devices_inference_times")

        largest_graph_size = df_forward_pass["size"].max()
        df_largest_graphs = df_forward_pass[df_forward_pass["size"] == largest_graph_size ]
        fastest_device = df_largest_graphs[
            df_largest_graphs["forward_pass_s"] == df_largest_graphs["forward_pass_s"].min()]["device"].item()

        d = next(iter(ErdosRenyiGenerator(largest_graph_size, hamilton_existence_prob)))
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("inference"):
                operation_forward_step(HamS_model, torch_g.data.Batch.from_data_list([d]))
        print(f"Profiling EPD model forward pass on graphs of size {largest_graph_size}")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=profiler_row_limit))

        d = next(iter(ErdosRenyiGenerator(largest_graph_size, hamilton_existence_prob)))
        runtime_records = profile_EPD(HamS_model, torch_g.data.Batch.from_data_list([d]))
        df_component_runtimes = pandas.DataFrame.from_records(runtime_records)
        fig, ax = plt.subplots()
        seaborn.barplot(x="component", y="runtime", data=df_component_runtimes, ax=ax).set_title("Component runtimes")
        fig.savefig(figure_output_path / "component runtimes")

        runtime_comparison_sizes = [50, 100, 150, 250, 500, 1000, 2000, 4000, 8000]
        runtime_nr_examples_per_size = 2
        records = []
        for s in runtime_comparison_sizes:
            graph_generator = ErdosRenyiGenerator(s, hamilton_existence_prob)
            for d in itertools.islice(graph_generator, runtime_nr_examples_per_size):
                HamS_runtime, [(HamS_path_tensor, HamS_probabilites)] = time_operation(
                    lambda a: HamS_model.batch_run_greedy_neighbor(a),
                    [torch_g.data.batch.Batch.from_data_list([d])], HamS_model.device)
                nn_path_len = HamS_path_tensor[HamS_path_tensor != -1].unique().shape[0]
                records.append({"description": "HamS", "graph_size": s, "runtime": HamS_runtime,
                             "path_size": nn_path_len})

                try:
                    concorde_solver = ConcordeHamiltonSolver()
                    concorde_path, concorde_process_time, _ = concorde_solver.time_execution(d)
                    records.append({"description": "Concorde", "graph_size": s, "runtime": concorde_process_time,
                                "path_size": s})
                except Exception as ex:
                    print("Concorde not installed on the system")
                    raise ex

                for heuristic, description in zip([least_degree_first_heuristics, HybridHam],
                                                  ["least_degree_first", "HybridHam"]):
                    runtime, path = time_operation(lambda a: heuristic(a.num_nodes, a.edge_index), [d], "cpu")
                    records.append({"description": description, "graph_size": s,
                                "runtime": runtime, "path_size": len(path)})

        df_heuristics_comparison = pandas.DataFrame.from_records(records)
        df_heuristics_comparison["estimated_runtime"] \
            = df_heuristics_comparison["runtime"] * df_heuristics_comparison["graph_size"] / df_heuristics_comparison["path_size"]
        fig, ax = plt.subplots()
        seaborn.lineplot(x="graph_size", y="estimated_runtime", hue="description", data=
            df_heuristics_comparison.groupby(by=["description", "graph_size"]).mean().reset_index(), ax=ax)\
            .set_title("Runtime comparison")
        fig.savefig(figure_output_path / "Heuristics runtime comparison")
