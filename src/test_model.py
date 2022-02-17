import copy

from pytorch_lightning import Trainer
import pytorch_lightning

import src.constants as constants
import src.Models as Models
import src.data.GraphDataset as GraphDataset
from src.data.InMemoryDataset import ErdosRenyiInMemoryDataset

def test_model_unified(model):
    dataset = ErdosRenyiInMemoryDataset(constants.EVALUATION_DATA_FOLDERS)
    sizes = list(set(graph_item.graph.num_nodes for graph_item in dataset))
    test_dataloaders = []
    for size in sizes:
        subset_dataset = copy.deepcopy(dataset)
        subset_dataset.data_list = \
            [data_item for data_item in dataset.data_list if data_item[ErdosRenyiInMemoryDataset.NUM_NODES_TAG] == size]
        test_dataloaders.append(GraphDataset.GraphDataLoader(subset_dataset))

    results = {}
    trainer = pytorch_lightning.Trainer()
    for size, test_dataloader in zip(size, test_dataloaders):
         results[size] = trainer.test(model, dataloaders=test_dataloaders[1])
    return results


if __name__ == "__main__":
    model_class = Models.EncodeProcessDecodeAlgorithm
    model_checkpoint = None
    model = model_class.load_from_checkpoint(model_checkpoint)
    results = test_model_unified(model)
    print(results)