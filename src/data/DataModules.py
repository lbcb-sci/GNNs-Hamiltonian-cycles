import pytorch_lightning as torch_lightning

class ArtificialCycleDataModule(torch_lightning.LighntingDataModule):
    def __init__(self, test_examples_root_dir="DATA", ham_existance_percentage=0.8):
        super(ArtificialCycleDataModule, )
        self.test_examples_root_dir = test_examples_root_dir
        self.ham_existance_percentage = ham_existance_percentage

    def prepare_data():
        pass

    def
