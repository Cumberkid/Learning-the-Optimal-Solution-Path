import torch
from torch import nn

# this initializes with random weights. Need to either set a seed or force initialization somewhere for reproducibility.
# automatically fits an intercept. To turn off intercept, set bias=False in nn.Linear()
class Portfolio_Allocation(nn.Module):
    def __init__(self, input_dim, output_dim, reg_param, init_weight, init_intercept):
        super(Portfolio_Allocation, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        # self.actv = nn.Softplus()
        self.reg_param = reg_param
        
        # initialize for better performance
        with torch.no_grad():
          self.linear.weight.copy_(init_weight)
          self.linear.bias.data.fill_(init_intercept)
          
    def forward(self, x):
        # return self.actv(self.linear(x))
        return self.linear(x)
        
    def penalty(self):
        theta = self.linear.weight.clone().detach()
        theta_0 = self.linear.bias.clone().detach()
        mu = 0.01
        return torch.sum(torch.sqrt(theta**2 + mu**2) - mu) + (torch.sqrt(theta_0**2 + mu**2) - mu)
