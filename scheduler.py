import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, total_iters, min_lr=0.0, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.min_lr = min_lr
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1  # step starts from 1
        if step < self.warmup_iters:
            # Linear warmup
            return [base_lr * step / self.warmup_iters for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            cosine_step = step - self.warmup_iters
            cosine_total = self.total_iters - self.warmup_iters
            return [
                self.min_lr + 0.5 * (base_lr - self.min_lr) *
                (1 + math.cos(math.pi * cosine_step / cosine_total))
                for base_lr in self.base_lrs
            ]
