import numpy as np

def weigh(loss_dict, weights):
    unweighted_loss_dict = dict()
    for loss_name, loss_item in loss_dict.items():
        unweighted_loss_dict[loss_name] = loss_item.clone().detach()
        if loss_name in weights:
            loss_dict[loss_name] = loss_item * weights[loss_name]
    
    return loss_dict, unweighted_loss_dict

# KL annealing based on https://arxiv.org/abs/1511.06349
def beta(iteration, milestone, eps=1e-4):
    # Use eps~1e-04 to constrain kl divergence from diverging before beta kicks in
    if milestone > 0:
        return 1 / (1 + np.exp(-0.01*(iteration - (milestone+500)))) + eps
    else:
        return 1.0