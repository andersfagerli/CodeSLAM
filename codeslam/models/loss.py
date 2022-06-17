import torch
import torch.nn as nn
import torch.nn.functional as F

class Criterion(nn.Module):
    def __init__(self, cfg):
        super(Criterion, self).__init__()
    
    def kl_divergence(self, mu, logvar):
        loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
        return loss
    
    def reconstruction_loss(self, input, target, b=None, eps=1e-06):
        if b is not None:
            b_clamped = torch.clamp(b, min=eps)
            loss = torch.mean(torch.mean(F.l1_loss(input, target, reduction='none') / b_clamped + torch.log(b_clamped), dim=(2,3)), dim=(0,1))
        else:
            loss = F.l1_loss(input, target)
    
        return loss
    
    def forward(self, depth, mu, logvar, target, b=None):
        kl_div = self.kl_divergence(mu, logvar)
        reconstruction_loss = self.reconstruction_loss(depth, target, b)

        loss_dict = dict(
            kl_div=kl_div,
            recon_loss=reconstruction_loss,
        )

        return loss_dict