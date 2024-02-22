import torch.optim as optim

class AlphaT_MinLR_Scheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, init_lr, alpha):
        self.init_lr = init_lr
        self.alpha = alpha
        super().__init__(optimizer)

    def get_lr(self):
        return [min(self.init_lr, self.alpha / (self.last_epoch + 1)) for _ in self.optimizer.param_groups]
