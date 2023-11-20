# -*- coding: utf-8 -*-
"""SGD.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1k2lvuMpxc4ZNXM0WIrgsPplotz09ZHP0
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

import random
import math
from scipy.special import legendre

# compute \Phi(\lambda)
def phi_lam_Legendre(lam, basis_dim):
    # Transform the lam to [-1, 1] interval
    lam_transformed = 2 * lam - 1
    vec = torch.zeros(basis_dim)
    for i in range(basis_dim):
        vec[i] = math.sqrt(2*i+1) * legendre(i)(lam_transformed)
    return vec.to(device)

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
    def forward(self, lam):
        phi = self.basis_fn(lam, self.basis_dim)
        return self.linear(phi)


# trace_frequency is measured in number of batches. -1 means don't print
def train_SGD(dataloader, model, loss_fn, optimizer, distribution='uniform', trace_frequency=-1):
    model.train()
    actv = nn.Sigmoid()
    for batch, (X_train, y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)

        rndm_lam = torch.tensor(0.5)
        # SGD picks random regulation parameter lambda
        if distribution == 'uniform':
            rndm_lam = torch.torch.distributions.Uniform(0, 1).sample()
        # print(f"random lam = {rndm_lam}")

        # Compute predicted y_hat
        theta = model(rndm_lam.cpu())
        pred = torch.mm(X_train, theta[1:].view(-1, 1))
        if model.intercept:
            const = torch.ones(len(X_train), 1).to(device)
            pred += torch.mm(const, theta[0].view(-1, 1))
        pred = actv(pred)
        # print(theta[0])

        loss = (1 - rndm_lam) * loss_fn(pred.view(-1, 1), y_train.view(-1, 1))
        loss += rndm_lam * 0.5 * theta.norm(p=2)**2

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (trace_frequency > 0) & (batch % trace_frequency == 0):
        #     loss, current = loss.item(), (batch + 1) * len(X_train)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Test function
def test_SGD(dataloader, model, loss_fn, lam):
    model.eval() #important
    actv = nn.Sigmoid()
    with torch.no_grad():  #makes sure we don't corrupt gradients and is faster
        for batch, (X_test, y_test) in enumerate(dataloader):
          X_test, y_test = X_test.to(device), y_test.to(device)

          # Compute prediction error
          theta = model(lam)
          pred = actv(torch.mm(X_test, theta[1:].view(-1, 1)) + theta[0].item())
          # print(f"prediction = {pred}")

          oos = (1 - lam) * loss_fn(pred.view(-1, 1), y_test.view(-1, 1))
          oos += lam * 0.5 * theta.norm(p=2)**2

    return oos.item()
    
# return a list of loss computed on a specified grid over the solution path
def get_losses_SGD(model, lam_min, lam_max, num_grid, data_loader, criterion):
    delta_lam = (lam_max - lam_min)/num_grid
    lambdas = torch.arange(lam_max, lam_min, (-1)*delta_lam)
    losses = []
    for lam in lambdas:
        losses.append(my_module.test_SGD(data_loader, model, criterion, lam))

    return losses

# return the supremum absolute error compared to the true loss accross the solution path    
def get_sup_error_SGD(lam_min, lam_max, true_loss_list, model, data_loader, criterion):
    fine_delta_lam = (lam_max - lam_min)/len(true_loss_list)
    # check sup error
    sup_error = 0
    for i in range(len(true_loss_list)):
        exact_soln = true_loss_list[i]
        temp = 1 - i * fine_delta_lam
        # approximate solution uses the linear weight of coarse grid model to test for regression parameter of the fine grid
        approx_soln = my_module.test_SGD(data_loader, model, criterion, temp)
        sup_error = torch.max(torch.tensor([sup_error, approx_soln - exact_soln]))
        # print(sup_error)
    return sup_error.item()