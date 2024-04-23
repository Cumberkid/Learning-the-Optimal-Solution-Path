import torch.optim.lr_scheduler as lr_scheduler

class AlphaT_MinLR_Scheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, init_lr, alpha):
        self.init_lr = init_lr
        self.alpha = alpha
        super().__init__(optimizer)

    def get_lr(self):
        return [min(self.init_lr, self.alpha / (self.last_epoch + 1)) for _ in self.optimizer.param_groups]


class CustomScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, gamma=0.97, last_epoch=-1):
        self.factor = gamma
        super(CustomScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.factor for base_lr in self.base_lrs]
