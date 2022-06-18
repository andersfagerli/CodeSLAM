from torch.optim.lr_scheduler import _LRScheduler
from bisect import bisect_right

# Based on https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py

class LinearMultiStepWarmUp(_LRScheduler):
    def __init__(self, cfg, optimizer, milestones=None, last_epoch=-1):
        self.gamma = cfg.TRAINER.OPTIMIZER.GAMMA
        self.milestones = cfg.TRAINER.OPTIMIZER.MILESTONES if milestones is None else milestones
        self.warmup_period = cfg.TRAINER.OPTIMIZER.WARMUP_PERIOD

        assert self.gamma > 0, f'cfg.OPTIMIZER.GAMMA must be > 0 for learning rate scheduling'

        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.warmup_period > 0.0:
            warmup_factor = min(1.0, (self._step_count+1)/self.warmup_period)
        else:
            warmup_factor = 1.0

        return [base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs]


