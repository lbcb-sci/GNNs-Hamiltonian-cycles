import torch
from pytorch_lightning.callbacks import LambdaCallback


def _measure_max_lp_norm(tensors, p):
    with torch.no_grad():
        norms = []
        for tensor in tensors:
            norms.append(torch.linalg.vector_norm(tensor, p))
        return torch.stack(norms).max()


def _measure_logits_max_p_norm(p, pl_trainer, pl_module, *args, **kwargs):
    if "logits_per_step" in vars(pl_module):
        pl_module.log(f"logits/max_l{p}", _measure_max_lp_norm(pl_module.logits_per_step, p))


def _measure_weights_max_p_norm(p, pl_trainer, pl_module, *args, **kwargs):
    pl_module.log(f"weights/max_l{p}",
                  _measure_max_lp_norm((param for _, param in pl_module.named_parameters()), p)
                  )


def _get_gradients_if_all_exist(pl_module):
    grad_list = list(param.grad for _, param in pl_module.named_parameters())
    if any([grad is None for grad in grad_list]):
        return None
    else:
        return grad_list


def _measure_grad_max_lp_norm(p, pl_trainer, pl_module, *args, **kwargs):
    grad_list = _get_gradients_if_all_exist(pl_module)
    if grad_list:
        max_lp = _measure_max_lp_norm(grad_list, p)
    else:
        max_lp = -1.
    pl_module.log(f"gradients/max_l{p}", max_lp)



def _measure_total_flat_grad(p, pl_trainer, pl_module, *args, **kwargs):
    grad_list = _get_gradients_if_all_exist(pl_module)
    if grad_list:
        total_flat_grad = torch.concat([param.grad.flatten() for _, param in pl_module.named_parameters()])
        flat_lp = torch.linalg.vector_norm(total_flat_grad, p)
    else:
        flat_lp = -1.
    pl_module.log(f"flat_grad/l{p}", flat_lp)


def create_lp_callback(target_type, p_norm=2):
    if target_type == "max_lp_logits":
        _fn = _measure_logits_max_p_norm
    elif target_type == "max_lp_weights":
        _fn = _measure_weights_max_p_norm
    elif target_type == "max_lp_gradients":
        _fn = _measure_grad_max_lp_norm
    elif target_type == "flat_lp_gradients":
        _fn = _measure_total_flat_grad

    return LambdaCallback(
        on_before_zero_grad=lambda *args, **kwags: _fn(p_norm, *args, **kwags)
    )
