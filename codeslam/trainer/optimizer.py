import torch

def make_optimizer(cfg, model):
    if cfg.TRAINER.OPTIMIZER.TYPE == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.TRAINER.OPTIMIZER.LR,
            weight_decay=cfg.TRAINER.OPTIMIZER.WEIGHT_DECAY
        )
    elif cfg.TRAINER.OPTIMIZER.TYPE == "sgd":    
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.TRAINER.OPTIMIZER.LR,
            weight_decay=cfg.TRAINER.OPTIMIZER.WEIGHT_DECAY,
            momentum=cfg.TRAINER.OPTIMIZER.MOMENTUM
        )
    else:
        raise RuntimeError("Optimizer \"{}\" in config is not supported, check optimizer.py for the supported optimizers".format(cfg.OPTIMIZER.TYPE))
    
    return optimizer