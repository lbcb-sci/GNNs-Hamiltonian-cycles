import torch

from hamgnn.data.InMemoryDataset import ErdosRenyiInMemoryDataset
from hamgnn.constants import \
    GRAPH_DATA_DIRECTORY_SIZE_GENERALISATION,\
    GRAPH_DATA_DIRECTORY_HAM_PROB_GENERALISATION,\
    GRAPH_DATA_DIRECTORY_SUPERCRITICAL_CASE,\
    RESULTS_SIZE_GENERALIZATION_PLOT_DATASET_SIZES,\
    DATASET_SIZE_ACCURACY_VARIANCE_002_95PERC_CONF,\
    DEFAULT_HAMILTONIAN_PROBABILITY,\
    RESULTS_P_PARAM_GENERALIZATION_PLOT_HAM_PROBABILITY,\
    RESULTS_SUPERCRITICAL_REGIME_ER_P_PARAM

def create_dataset_in_critical_regime(store_directory, sizes, nr_examples, hamilton_existence_probabilities):
    if not store_directory.exists():
        store_directory.mkdir(parents=True)
    else:
        if len([x for x in store_directory.iterdir() if x.suffix == ".pt"]) > 0:
            print(f"Found existing files data in {store_directory}. Skipping this folder")
            return
    print(f"Generating graphs for following (size, nr_examples, hamilton_existence_probabilty) parameters:\n"
          f"{list(zip(sizes, nr_examples, hamilton_existence_probabilities))}.\n"
          f"Data will be stored in {store_directory}. This might take a while...")
    ErdosRenyiInMemoryDataset.create_dataset_in_critical_regime(
        out_folder=store_directory,
        sizes=sizes,
        nr_examples=nr_examples,
        hamilton_existence_prob=hamilton_existence_probabilities,
        solve_with_concorde=True)


if __name__ == '__main__':
    torch.set_num_threads(1)

    print(f"Crating dataset of Erdos-Renyi graphs for test generalisation to different sizes.")
    create_dataset_in_critical_regime(
        store_directory=GRAPH_DATA_DIRECTORY_SIZE_GENERALISATION,
        sizes=RESULTS_SIZE_GENERALIZATION_PLOT_DATASET_SIZES,
        nr_examples=[DATASET_SIZE_ACCURACY_VARIANCE_002_95PERC_CONF for _ in RESULTS_SIZE_GENERALIZATION_PLOT_DATASET_SIZES],
        hamilton_existence_probabilities=[DEFAULT_HAMILTONIAN_PROBABILITY for _ in RESULTS_SIZE_GENERALIZATION_PLOT_DATASET_SIZES]
        )
    print(f"Completed")

    print(f"Creating dataset of Erdos-Renyi graphs for testing generalisation to different Hamiltonian cycle existence probabilities")
    create_dataset_in_critical_regime(
        store_directory=GRAPH_DATA_DIRECTORY_HAM_PROB_GENERALISATION,
        sizes=[RESULTS_SIZE_GENERALIZATION_PLOT_DATASET_SIZES[0] for _ in RESULTS_P_PARAM_GENERALIZATION_PLOT_HAM_PROBABILITY],
        nr_examples=[DATASET_SIZE_ACCURACY_VARIANCE_002_95PERC_CONF for _ in RESULTS_P_PARAM_GENERALIZATION_PLOT_HAM_PROBABILITY],
        hamilton_existence_probabilities=RESULTS_P_PARAM_GENERALIZATION_PLOT_HAM_PROBABILITY
    )
    print(f"Completed.")

    print(f"Creating dataset of Erdos-Renyi graphs in supercritical regime")
    if not GRAPH_DATA_DIRECTORY_SUPERCRITICAL_CASE.exists():
        GRAPH_DATA_DIRECTORY_SUPERCRITICAL_CASE.mkdir(parents=True)
    elif len([x for x in GRAPH_DATA_DIRECTORY_SUPERCRITICAL_CASE.iterdir() if x.suffix == ".pt"]) > 0:
            print(f"Found exsiting graph data in {GRAPH_DATA_DIRECTORY_SUPERCRITICAL_CASE}. Skipping this folder")
    else:
        ErdosRenyiInMemoryDataset.create_dataset_from_edge_probabilities(
            out_folder=GRAPH_DATA_DIRECTORY_SUPERCRITICAL_CASE,
            sizes=RESULTS_SIZE_GENERALIZATION_PLOT_DATASET_SIZES,
            nr_examples=[DATASET_SIZE_ACCURACY_VARIANCE_002_95PERC_CONF for _ in RESULTS_SIZE_GENERALIZATION_PLOT_DATASET_SIZES],
            edge_existence_probability=[RESULTS_SUPERCRITICAL_REGIME_ER_P_PARAM for _ in RESULTS_SIZE_GENERALIZATION_PLOT_DATASET_SIZES],
        )
        print(f"Completed.")
