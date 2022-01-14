import torch
import torch.nn.functional as F

def kl_divergence(mu, logvar):
    loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
    return loss

def reconstruction_loss(input, target, b, eps=1e-06):
    b_clamped = torch.clamp(b, min=eps)
    loss = torch.mean(torch.mean(F.l1_loss(input, target, reduction='none') / b_clamped + torch.log(b_clamped), dim=(2,3)), dim=(0,1))
    loss_unweighted = F.l1_loss(input, target)
 
    return loss, loss_unweighted

