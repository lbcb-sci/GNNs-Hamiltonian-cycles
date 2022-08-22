from operator import index
import torch.optim.lr_scheduler as lr_scheduler

def _polylinear_function(x, transitions_x, transitions_y):
    all_smaller = [index for index in range(len(transitions_x)) if transitions_x[index] <= x]
    if len(all_smaller) == 0:
        return transitions_y[0]
    elif len(all_smaller) == len(transitions_x):
        return transitions_y[-1]
    index = all_smaller[-1]
    return (transitions_y[index] * (transitions_x[index + 1] - x) + transitions_y[index + 1] * (x - transitions_x[index])) \
        / (transitions_x[index + 1] - transitions_x[index])


def build_linear_warmup_linear_decay_scheduler_lambda(warmup_start_lr, warmup_end_lr, final_lr, warmup_epochs, total_epochs):
    return lambda epoch: _polylinear_function(epoch, [0, warmup_epochs, total_epochs], [warmup_start_lr, warmup_end_lr, final_lr])
