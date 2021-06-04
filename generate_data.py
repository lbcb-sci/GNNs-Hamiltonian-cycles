from src.DatasetBuilder import ErdosRenyiInMemoryDataset
from src.constants import EVALUATION_DATA_FOLDERS, DEFAULT_DATASET_SIZES, DEFAULT_EXAMPLES_PER_SIZE_IN_DATASET,\
    HAMILTONIAN_PROBABILITY

if __name__ == '__main__':
    assert len(EVALUATION_DATA_FOLDERS)
    data_folder = EVALUATION_DATA_FOLDERS[0]
    sizes = DEFAULT_DATASET_SIZES
    nr_examples = DEFAULT_EXAMPLES_PER_SIZE_IN_DATASET
    print(f"Crating dataset of Erdos-Renyi graphs of sizes {sizes} with {DEFAULT_EXAMPLES_PER_SIZE_IN_DATASET}"
          f" examples per size. This might take a while ...")
    ErdosRenyiInMemoryDataset.create_dataset(data_folder, sizes, DEFAULT_EXAMPLES_PER_SIZE_IN_DATASET,
                                             HAMILTONIAN_PROBABILITY, solve_with_concorde=True)
    print(f"Successfully created Erdos-Renyi dataset with sizes {sizes} of {nr_examples} examples each.")
