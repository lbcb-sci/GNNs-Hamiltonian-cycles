import torch
from torch.nn import functional as F
from pytorch_lightning.callbacks import LambdaCallback


def _measure_max_logits_p_norm(pl_trainer, pl_module, p=2):
    if "logits_per_step" in vars(pl_module):
        norms = []
        for l in pl_module.logits_per_step:
            norms.append(F.normalize(l, p))
        pl_trainer.log(f"max_logits_l{p}_norm", torch.stack(norms).max())

callback_max_logits_2_norm = LambdaCallback(
    on_after_backward=_measure_max_logits_p_norm
)

def _measure_max_weights_p_norm(pl_trainer, pl_module, p=2):
    norms = []
    for name, param in pl_module.named_parameters():
        norms.append(F.normalize(param, p))
    pl_trainer.log(f"max_l2_of_all_layers", torch.stack(norms).max())


callback_max_weights_2_norm = LambdaCallback(
    on_after_backward= _measure_max_weights_p_norm
)