import pytorch_lightning as torch_lightning

class BaseGraphDatamodule(torch_lightning.LightningDataModule):
    def __init__(self, train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)

    def train_dataloader(self):
        return super().train_dataloader()

class ArtificialCycleDataModule(torch_lightning.LighntingDataModule):
    def __init__(self, test_examples_root_dir="DATA", ham_existance_percentage=0.8):
        super(ArtificialCycleDataModule, )
        self.test_examples_root_dir = test_examples_root_dir
        self.ham_existance_percentage = ham_existance_percentage

    def prepare_data():
        pass

    def
