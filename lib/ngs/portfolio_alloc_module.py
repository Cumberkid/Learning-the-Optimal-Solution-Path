import torch
from torch import nn

# this initializes with random weights. Need to either set a seed or force initialization somewhere for reproducibility.
# automatically fits an intercept. To turn off intercept, set bias=False in nn.Linear()
class Portfolio_Allocation(nn.Module):
    def __init__(self, input_dim, output_dim, hyper_param, init_weight):
        super(Portfolio_Allocation, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        # self.actv = nn.Softplus()
        self.hyper_param = hyper_param
        
        # initialize for better performance
        with torch.no_grad():
          self.linear.weight.copy_(init_weight)
          
    def forward(self, x):
        # return self.actv(self.linear(x))
        return self.linear(x)

