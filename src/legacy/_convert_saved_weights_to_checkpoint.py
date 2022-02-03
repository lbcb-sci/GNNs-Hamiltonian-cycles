# %%
from src.Models import EncodeProcessDecodeAlgorithm, GatedGCNEmbedAndProcess

import torch
# %%
HamS = EncodeProcessDecodeAlgorithm(True, processor_depth=5, hidden_dim=32, loss_type="entropy")
HamR = GatedGCNEmbedAndProcess(True, embedding_depth=8, processor_depth=5, hidden_dim=32, loss_type="entropy")

# %%
import pytorch_lightning
from src.data.DataModules import ArtificialCycleDataModule

datamodule = ArtificialCycleDataModule()
trainer = pytorch_lightning.Trainer(max_epochs=0, num_sanity_val_steps=0)
trainer.fit(HamS, datamodule=datamodule)
trainer.save_checkpoint("new_weights/lightning_HamS.ckpt")
lightning_HamS = EncodeProcessDecodeAlgorithm.load_from_checkpoint("new_weights/lightning_HamS.ckpt")

trainer = pytorch_lightning.Trainer(max_epochs=0, num_sanity_val_steps=0)
trainer.fit(HamR, datamodule=datamodule)
trainer.save_checkpoint("new_weights/lightning_HamR.ckpt")
lightning_HamR = GatedGCNEmbedAndProcess.load_from_checkpoint("new_weights/lightning_HamR.ckpt")

print(f"Lightning HamS hyperparameters are {lightning_HamS.hparams}")
print(f"Lightning HamR hyperparameters are {lightning_HamR.hparams}")

print("Checking that all parameters of newly created lightning modules are identical to original")
for original, lighting in [(HamS, lightning_HamS), (HamR, lightning_HamR)]:
    for p1, p2 in zip(original.parameters(), lighting.parameters()):
        assert torch.all(torch.isclose(p1, p2))
print("Check complete! Models are identical!")

# %%
