import numpy as np
import torch.nn.functional as F
import torch

# KL annealing based on https://arxiv.org/abs/1511.06349
def beta(iteration, milestone, eps=0.0):
    # Use eps~1e-03 to constrain kl divergence from diverging before beta kicks in
    if milestone > 0:
        return 1 / (1 + np.exp(-0.01*(iteration - milestone))) + eps
    else:
        return 1.0

# Based on https://arxiv.org/abs/2002.07514
def gamma(input, target, prev_gamma=None):
    mse = F.mse_loss(input, target)
    rmse = 0.0 if mse < 1e-06 else torch.sqrt(mse)
    if not prev_gamma:
        return rmse
    else:
        return min(prev_gamma, rmse)