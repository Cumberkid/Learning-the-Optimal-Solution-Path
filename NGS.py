# -*- coding: utf-8 -*-
"""NGS.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1N4WT2GmNBrDo0KCGBJ1mUmdJa9yyYyH8

# Import the necessary libraries
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader  #for creating the dataset


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

"""# Running Logistic Regression Through Tourch NN"""

#Prep data for Pytorch Dataloader
class Regression_Data(Dataset):
    def __init__(self, X, y):
        self.X = X  if isinstance(X, torch.Tensor) else torch.FloatTensor(X)
        self.y = y  if isinstance(y, torch.Tensor) else torch.FloatTensor(y)
        self.input_dim = self.X.shape[1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx]

    def get_y(self):
        return self.y

# this initializes with random weights. Need to either set a seed or force initialization somewhere for reproducibility.
# automatically fits an intercept. To turn off intercept, set bias=False in nn.Linear()
class Logistic_Regression(nn.Module):
    def __init__(self, input_dim, output_dim, reg_param, init_weight):
        super(Logistic_Regression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.actv = nn.Sigmoid()
        self.reg_param = reg_param

        # initialize for better performance
        with torch.no_grad():
          self.linear.weight.copy_(init_weight)
          self.linear.bias.data.fill_(0)

    def forward(self, x):
        return self.actv(self.linear(x))

    def ridge_term(self):
        return self.linear.weight.norm(p=2)**2 + self.linear.bias.norm(p=2)**2

"""The "train" function executes optimization on the input dataset w.r.t. the input loss function with the input optimizer. We will use the pytorch built-in SGD optimizer later, but note that this optimizer is actually just a deterministic gradient descent program.

To randomize for SGD, we notice that the loss function is a sum of losses of all training data points, and a standard SGD would randomly choose one of those points to descend on at each step of descent.

To speed up, we use a batch of data points to replace a single data point at each step of descent. When batch size = 1, this is equivalent to a standard SGD; and when batch size = training set size, this is simply a deterministic gradient descent.
"""

# trace_frequency is measured in number of batches. -1 means don't print
def train(dataloader, model, loss_fn, optimizer, trace_frequency = -1):
    # size = len(dataloader.dataset)
    model.train()
    # here, the "batch" notion takes care of randomization
    for batch, (X_train, y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)
        # print(batch, len(X_train))

        # Compute predicted y_hat
        pred = model(X_train)
        # With regularization
        loss = (1 - model.reg_param) * loss_fn(pred.view(-1, 1), y_train.view(-1, 1))
        loss += model.reg_param * 0.5 * model.ridge_term()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (trace_frequency > 0) & (batch % trace_frequency == 0):
        #     loss, current = loss.item(), (batch + 1) * len(X_train)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

"""The "test" function defined here is our objective function $h(\theta, \lambda) = (1-\lambda)BCE(X\theta, y) + \frac{\lambda}{2}\|\theta\|^2$. The linear weight from the above trained model is our $\theta$."""

# Test function
def test(dataloader, model, loss_fn, lam):
    model.eval() # important
    with torch.no_grad():  # makes sure we don't corrupt gradients and is faster
        for batch, (X_test, y_test) in enumerate(dataloader):
            X_test, y_test = X_test.to(device), y_test.to(device)

            # Compute prediction error
            pred = model(X_test)

            oos = (1 - lam) * loss_fn(pred.view(-1, 1), y_test.view(-1, 1))
            oos += lam * 0.5 * model.ridge_term()

    return oos.item()

"""# Naive Grid Search"""

# running gradient descent with fixed learning rate on a single grid point, i.e. for one specified lambda
def GD_on_a_grid(lam, epochs, weight, loss_fn, trainDataLoader, data_input_dim,
                 lr=1e-3, alpha=1, SGD=False, testDataLoader=None,
                 true_loss_list=None, fine_delta_lam=None, stopping_criterion=None):
    model = Logistic_Regression(data_input_dim, 1, lam, weight).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()

    if true_loss_list is not None:
        # exact solution
        i = torch.round((1-lam) / fine_delta_lam).int()
        if i >= len(true_loss_list):
            i -= 1
            i.int()
        exact_soln = true_loss_list[i]
        lam = 1 - i * fine_delta_lam
        # model.reg_param = lam
        # print(i)

    early_stop = False
    itr = 0
    for t in range(epochs):
        if SGD:
            # shrink learning rate
            lr = torch.min(torch.tensor([0.1, alpha/(t+1)]))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        train(trainDataLoader, model, loss_fn, optimizer, trace_frequency=5)
        if true_loss_list is not None:
            if (t+1) % 10 == 0:
                # do an accuracy check
                approx_soln = test(testDataLoader, model, loss_fn, lam)
                error = approx_soln - exact_soln
                # stopping criterion
                if error <= stopping_criterion:
                    itr += (t+1)
                    early_stop = True
                    break  # Early stop

    if not early_stop:
        itr += epochs

    return model, itr

"""Naive Grid Search starts from $\lambda = 1$ and decreases $\lambda$ by $\Delta\lambda = \frac{\lambda_\text{max} - \lambda_\text{min}}{\text{# of grid}}$. The model trained on each grid point $(\lambda - \Delta\lambda)$ initializes weight with the linear weight of the model trained on the previous grid point $\lambda$."""

# do the whole naive grid search over a list of uniformly spaced lambda's
# from lam_min to lam_max
# returns a list of trained models
def naive_grid_search(lam_min, lam_max, num_grid, epochs, loss_fn, trainDataLoader,
                      data_input_dim, lr=1e-3, alpha=1, SGD=False,
                      testDataLoader=None, true_loss_list=None, stopping_criterion=None):
    delta_lam = (lam_max - lam_min)/num_grid
    fine_delta_lam = None
    if true_loss_list is not None:
        fine_delta_lam = (lam_max - lam_min)/len(true_loss_list)
    model_list = []
    total_itr = 0
    # create a list of lambda's
    lambdas = torch.arange(lam_max, lam_min, (-1)*delta_lam).to(device)

    # first weight is initialized at 0
    weight = torch.zeros(data_input_dim)

    for lam in lambdas:
        # print(f"Running model on lambda = {lam}")
        model, itr = GD_on_a_grid(lam, epochs, weight, loss_fn,
                                  trainDataLoader=trainDataLoader,
                                  data_input_dim=data_input_dim,
                                  lr=lr, alpha=alpha,
                                  SGD=SGD, testDataLoader=testDataLoader,
                                  true_loss_list=true_loss_list,
                                  fine_delta_lam=fine_delta_lam,
                                  stopping_criterion=stopping_criterion)
        weight = model.linear.weight
        model_list.append(model)
        total_itr += itr
        # print(total_itr)

    return model_list, total_itr

"""Helper function that takes in a list of coarse grid models and returns the sup error over $\lambda\in[0,1]$ compared to the exact solutions."""

def get_losses(coarse_model_list, data_loader, criterion):
  losses = []
  for model in coarse_model_list:
      losses.append(test(data_loader, model, criterion, model.reg_param))

  return losses

def get_sup_error(lam_min, lam_max, true_loss_list, coarse_model_list, data_loader, criterion):
    fine_delta_lam = torch.tensor((lam_max - lam_min)/len(true_loss_list))
    delta_lam = torch.tensor((lam_max - lam_min)/len(coarse_model_list))
    # check sup error
    sup_error = 0
    coarse_grid = 0
    for i in range(len(true_loss_list)):
        true_loss = true_loss_list[i]
        
        # coarse_grid = torch.round(i * fine_delta_lam / delta_lam).int()
        # if coarse_grid >= len(coarse_model_list):
        #     coarse_grid -= 1
        #     coarse_grid.int()
        # print(i, coarse_grid)
        lam = 1 - i * fine_delta_lam
        if (coarse_grid + 1) < len(coarse_model_list):
            if (coarse_model_list[coarse_grid].reg_param - lam) >= (lam - coarse_model_list[coarse_grid + 1].reg_param):
                coarse_grid += 1
        # approximate solution uses the linear weight of coarse grid model to test for regression parameter of the fine grid
        approx_loss = test(data_loader, coarse_model_list[coarse_grid], criterion, lam)
        sup_error = torch.max(torch.tensor([sup_error, approx_loss - true_loss]))
        # print(i, coarse_grid, sup_error)
    return sup_error.item()