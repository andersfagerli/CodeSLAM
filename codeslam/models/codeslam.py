import torch.nn as nn

from .unet import UNet
from .cvae import CVAE

class CodeSLAM(nn.Module):
    def __init__(self, cfg):
        super(CodeSLAM, self).__init__()
        self.unet = UNet(cfg)
        self.cvae = CVAE(cfg)
    
    def forward(self, image, target=None):
        feature_maps = self.unet(image)
        depth, mu, logvar = self.cvae(feature_maps, target)

        uncertainty = feature_maps[-1]

        return (depth, uncertainty, mu, logvar)
