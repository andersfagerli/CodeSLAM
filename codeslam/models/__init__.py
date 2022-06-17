from .unet import UNet
from .cvae import CVAE
from .codeslam import CodeSLAM

_MODELS = {
    'unet': UNet,
    'cvae': CVAE,
    'codeslam': CodeSLAM,
}

def make_model(cfg):
    if cfg.MODEL.NAME not in _MODELS:
        raise RuntimeError("Model \"{}\" in config is not supported, check models/__init__.py for the supported models".format(cfg.MODEL.NAME))
    
    model = _MODELS[cfg.MODEL.NAME](cfg)
    
    return model