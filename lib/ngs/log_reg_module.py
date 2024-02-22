import torch
from torch import nn

# this initializes with random weights. Need to either set a seed or force initialization somewhere for reproducibility.
# automatically fits an intercept. To turn off intercept, set bias=False in nn.Linear()
class Logistic_Regression(nn.Module):
    def __init__(self, input_dim, output_dim, reg_param, init_weight, init_intercept):
        super(Logistic_Regression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        # self.actv = nn.Sigmoid()
        self.reg_param = reg_param
        
        # initialize for better performance
        with torch.no_grad():
          self.linear.weight.copy_(init_weight)
          self.linear.bias.data.fill_(init_intercept)
          
    def forward(self, x):
        # return self.actv(self.linear(x))
        return self.linear(x)

    def criterion(self, output, target):
        criterion = torch.nn.BCEWithLogitsLoss()
        return criterion(output, target)
        
    def ridge_term(self):
        return self.linear.weight.norm(p=2)**2 + self.linear.bias.norm(p=2)**2
