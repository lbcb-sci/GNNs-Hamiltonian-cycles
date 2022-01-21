from torch.utils.data import DataLoader
import pytorch_lightning as torch_lightning

from src.Models import EncodeProcessDecodeAlgorithm
from src.data.InMemoryDataset import ErdosRenyiInMemoryDataset
from src.data.GraphGenerators import NoisyCycleGenerator
from src.data.GraphDataset import GraphDataLoader, GraphGeneratingDataset


if __name__ == "__main__":
    generator = NoisyCycleGenerator(50, 0.8)
    dataset = GraphGeneratingDataset(generator)
    dataloader = GraphDataLoader(dataset, batch_size=8)
    model = EncodeProcessDecodeAlgorithm(False)
    trainer = torch_lightning.Trainer()
    trainer.fit(model=model, train_dataloader=dataloader)
