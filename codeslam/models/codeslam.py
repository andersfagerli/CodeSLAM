import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .unet import UNet
from .cvae import CVAE
from .loss import Criterion

class CodeSLAM(nn.Module):
    def __init__(self, cfg):
        super(CodeSLAM, self).__init__()
        self.unet = UNet(cfg)
        self.cvae = CVAE(cfg)
        self.loss = Criterion(cfg)

        self._b: Tensor = None
        self._depth: Tensor = None
        self._target: Tensor = None
        self._unweighted_reconstruction_loss: float = None
    
    @property
    def b(self):
        return self._b
    
    @property
    def depth(self):
        return self._depth
    
    @property
    def unweighted_reconstruction_loss(self):
        assert self._depth is not None and self._target is not None,\
            f'Unweighted reconstruction loss is only accessible during training'

        return F.l1_loss(self._depth, self._target)
    
    def forward(self, input, target=None):
        is_training = target is not None

        # Forward
        feature_maps, b = self.unet(input)
        depth, mu, logvar = self.cvae(feature_maps, target)

        self._b = b[-1]
        self._depth = depth[-1]

        if is_training:
            self._target = target
            
            loss_dict = self.loss(depth, mu, logvar, target, b)
            return loss_dict
        else:
            result = dict(depth=self.depth, b=self.b)
            return result
