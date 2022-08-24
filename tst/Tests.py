import torch

from src import Models
import torch_geometric as torch_g
import os.path

from src.constants import WEIGHTS_BACKUP_PATH, EVALUATION_DATA_FOLDERS


def test_limiting_prob(graphs_directory):
    from src.DatasetBuilder import ErdosRenyiInMemoryDataset
    import os

    paths = [os.path.join(graphs_directory, p) for p in os.listdir(graphs_directory) if p.endswith(".pt")]
    paths.sort(key=lambda p: int(p.split(",")[0].split("(")[1]))
    sizes = [int(p.split(",")[0].split("(")[1]) for p in paths]
    print(f"Loading graphs from {paths}")

    datasets = [ErdosRenyiInMemoryDataset([p]) for p in paths]
    lengths = [len([x for x in d]) for d in datasets]
    print(f"Dataset lenghts are {lengths}")
    nr_hamiltonian = [len([x for x in data if x[1]]) for data in datasets]

    perc = [nr_hamiltonian[i] / lengths[i] for i in range(len(datasets))]
    print(f"Percentage of hamiltonian graphs per dataset: {perc}")
    print(sizes)


def test_model_on_dataset(nn_hamilton: Models.HamFinderGNN, data_directories):
    from src.Evaluation import EvaluationScores
    evals, sizes = EvaluationScores.evaluate_on_saved_data(nn_hamilton, 5000, data_directories)
    hamilton_perc, approx_hamilton_perc, full_walk_perc, long_walk_perc, perc_ham_graphs\
        = EvaluationScores.compute_accuracy_scores(evals, sizes)
    print(f"On graph of size {sizes} found Hamiltonian cycles in fractions of {hamilton_perc}.")
    print(f"Hamilton perc: {hamilton_perc}")


def _try_importing_encode_process_decode_model_from_database(weights_path, processor_depth):
    nn_hamilton = Models.EncodeProcessDecodeAlgorithm(False, processor_depth=processor_depth)
    nn_hamilton.load_weights(weights_path)
    return nn_hamilton


def _try_importing_embed_process_model_from_database(weights_path, embedding_depth):
    nn_hamilton = Models.GatedGCNEmbedAndProcess(False, embedding_depth=embedding_depth)
    nn_hamilton.load_weights(weights_path)
    return nn_hamilton


HamS_MODEL_ID = "1621492411869051554#140046741976896"
HamR_MODEL_ID = "1621631956874686418#140393951156032"


def import_HamS():
    weights_path = os.path.join(WEIGHTS_BACKUP_PATH, HamS_MODEL_ID)
    return _try_importing_encode_process_decode_model_from_database(weights_path, processor_depth=5)


def import_HamR():
    weights_path = os.path.join(WEIGHTS_BACKUP_PATH, HamR_MODEL_ID)
    return _try_importing_embed_process_model_from_database(weights_path, embedding_depth=8)


def avg_inference_time(inference_fn, data_path, inference_time_sample_size):
    import time
    from src.Evaluation import ErdosRenyiInMemoryDataset
    generator = ErdosRenyiInMemoryDataset(data_path)
    size_dict = {}
    for graph, hamilton_cycle in generator:
        if graph.num_nodes in size_dict and len(size_dict[graph.num_nodes]) < inference_time_sample_size:
            size_dict[graph.num_nodes] += [(graph, hamilton_cycle)]
        else:
            size_dict[graph.num_nodes] = [(graph, hamilton_cycle)]
    graph_sizes = [s for s in sorted(list(size_dict.keys()))]
    avg_inference_times = []
    for s in graph_sizes:
        total_time = 0
        nr_examples = 0
        for d, cycle in size_dict[s]:
            start_time = time.process_time()
            inference_fn(d)
            end_time = time.process_time()
            total_time += end_time - start_time
            nr_examples += 1
        avg_inference_times += [total_time/nr_examples]
    return avg_inference_times


def test_two_main_models(data_path=None, inference_time_sample_size=100):
    nn_hamilton_supervised = import_HamS()
    nn_hamilton_reinforcement = import_HamR()

    def supervised_inference_function(d):
        nn_hamilton_supervised.init_graph(d)
        nn_hamilton_supervised.batch_run_greedy_neighbor(torch_g.data.Batch.from_data_list([d]))

    def reinforcement_inference_function(d):
        nn_hamilton_reinforcement.init_graph(d)
        nn_hamilton_reinforcement.batch_run_greedy_neighbor(torch_g.data.Batch.from_data_list([d]))

    print(f"Testing supervised model {HamS_MODEL_ID}")
    test_model_on_dataset(nn_hamilton_supervised, data_path)
    print(f"Testing inference times...")
    avg_inference_supervised = avg_inference_time(supervised_inference_function, data_path, inference_time_sample_size)
    print(f"Avg. inference time per size: {avg_inference_supervised} s")
    print(f"Testing reinforcement model {HamR_MODEL_ID}")
    test_model_on_dataset(nn_hamilton_reinforcement, data_path)

    print(f"Testing inference times...")
    avg_inference_reinforcement = avg_inference_time(reinforcement_inference_function, data_path,
                                                     inference_time_sample_size)
    print(f"Avg. inference time per size: {avg_inference_reinforcement} s")


def test_concorde_inference_time(dataset=EVALUATION_DATA_FOLDERS, inference_time_sample_size=100):
    from src.Evaluation import ErdosRenyiInMemoryDataset
    import resource
    from src.ExactSolvers import ConcordeHamiltonSolver
    concorde = ConcordeHamiltonSolver()

    generator = ErdosRenyiInMemoryDataset(dataset)
    size_dict = {}
    for graph, hamilton_cycle in generator:
        if graph.num_nodes in size_dict and len(size_dict[graph.num_nodes]) < inference_time_sample_size:
            size_dict[graph.num_nodes] += [(graph, hamilton_cycle)]
        else:
            size_dict[graph.num_nodes] = [(graph, hamilton_cycle)]
    graph_sizes = [s for s in sorted(list(size_dict.keys()))]
    avg_inference_times = []
    for s in graph_sizes:
        print(f"Testing Concorde inference time on graphs of size {s} saved in {EVALUATION_DATA_FOLDERS}")
        total_time = 0
        nr_examples = 0
        for d, cycle in size_dict[s]:
            usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
            concorde.solve(d)
            usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
            total_time += usage_end.ru_utime - usage_start.ru_utime
            nr_examples += 1
        avg_inference_times += [total_time/nr_examples]
    print(f"Concorde evaluation times on graphs of size {graph_sizes}: {avg_inference_times}")


if __name__ == '__main__':
    torch.set_num_threads(1)
    data_folders = EVALUATION_DATA_FOLDERS
    test_two_main_models(data_folders)
    test_limiting_prob(data_folders[0])
    test_concorde_inference_time(data_folders)
