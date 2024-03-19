import torch
from torch import nn

# this initializes with random weights. Need to either set a seed or force initialization somewhere for reproducibility.
# automatically fits an intercept. To turn off intercept, set bias=False in nn.Linear()
class Basis_TF_SGD(nn.Module):
    def __init__(self, feature_dim, basis_dim, basis_fn, init_weight=None, intercept=True):
        super(Basis_TF_SGD, self).__init__()
        self.feature_dim = feature_dim
        self.basis_dim = basis_dim
        self.linear = nn.Linear(self.basis_dim, self.feature_dim + 1, bias=False)
        self.basis_fn = basis_fn
        self.intercept = intercept

        # initialize for better performance
        with torch.no_grad():
            if init_weight is not None:
                self.linear.weight.copy_(init_weight)
            else:
                self.linear.weight.data.fill_(0)
              
    # model takes input lambda and outputs theta
    def forward(self, lam, device='cpu'):
        phi = self.basis_fn(lam, self.basis_dim, device)
        return self.linear(phi)
