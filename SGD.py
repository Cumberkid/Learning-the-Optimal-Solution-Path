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
print(f"Using device: {device}")

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
    
# majority class is set to be class 1
# trace_frequency is measured in number of batches. -1 means don't print
def fair_train_SGD(dataloader, model, loss_fn, optimizer, distribution='uniform', trace_frequency=-1):
    model.train()
    actv = nn.Sigmoid()
    for batch, (X_train, y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)
        
        X_major = X_train[y_train == 1]
        y_major = torch.ones(len(X_major)).to(device)
        X_minor = X_train[y_train == 0]
        y_minor = torch.zeros(len(X_minor)).to(device)
        
        rndm_lam = torch.tensor(0.5)
        # SGD picks random regulation parameter lambda
        if distribution == 'uniform':
            rndm_lam = torch.torch.distributions.Uniform(0, 1).sample()
        # print(f"random lam = {rndm_lam}")

        # compute predicted y_hat
        theta = model(rndm_lam.cpu())
        # print(theta[0])
        pred_major = torch.mm(X_major, theta[1:].view(-1, 1))
        pred_minor = torch.mm(X_minor, theta[1:].view(-1, 1))
        if model.intercept:
            const_major = torch.ones(len(X_major), 1).to(device)
            pred_major += torch.mm(const_major, theta[0].view(-1, 1))
            const_minor = torch.ones(len(X_minor), 1).to(device)
            pred_minor += torch.mm(const_minor, theta[0].view(-1, 1))
        pred_major = actv(pred_major)
        pred_minor = actv(pred_minor)
        # fair loss function
        loss = (1 - rndm_lam) * loss_fn(pred_major.view(-1, 1), y_major.view(-1, 1)) 
        loss += rndm_lam * loss_fn(pred_minor.view(-1, 1), y_minor.view(-1, 1))
                
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # if (trace_frequency > 0) & (batch % trace_frequency == 0):
        #     loss, current = loss.item(), (batch + 1) * len(X_train)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# test function for fair objective
def fair_test_SGD(dataloader, model, loss_fn, lam):
    model.eval() #important
    actv = nn.Sigmoid()
    with torch.no_grad():  #makes sure we don't corrupt gradients and is faster
        for batch, (X_test, y_test) in enumerate(dataloader):
            X_test, y_test = X_test.to(device), y_test.to(device)
            
            X_major = X_test[y_test == 1]
            y_major = torch.ones(len(X_major)).to(device)
            X_minor = X_test[y_test == 0]
            y_minor = torch.zeros(len(X_minor)).to(device)
            
            # compute prediction error
            theta = model(lam)
            pred_major = actv(torch.mm(X_major, theta[1:].view(-1, 1)) + theta[0].item())
            pred_minor = actv(torch.mm(X_minor, theta[1:].view(-1, 1)) + theta[0].item())
            # print(f"prediction = {pred}")
            
            oos = (1 - lam) * loss_fn(pred_major.view(-1, 1), y_major.view(-1, 1)) 
            oos += lam * loss_fn(pred_minor.view(-1, 1), y_minor.view(-1, 1))
                    
    return oos.item()
    
def learn_optimal_solution_path(input_dim, basis_dim, phi_lam, epochs, dataLoader, loss_fn, lr=1e-3, alpha=1, init_lr=0.1, SGD=False, obj=None, intercept=True, trace_frequency=-1)
    if obj is None:
        print("Please enter the objective: 'logit' or 'fairness'")
        return
    # build the model
    model = SGD.Basis_TF_SGD(input_dim, basis_dim, phi_lam, intercept).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    sup_err_history = []
    num_itr_history = []

    for t in range(epochs):
        if SGD:
            # shrink learning rate
            lr = min([init_lr, alpha/(t+1)])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
        if obj == "logit":
            train_SGD(dataLoader, model, loss_fn, optimizer)
        elif obj == "fairness":
            fair_train_SGD(dataLoader, model, loss_fn, optimizer)

        if (t+1) % 100 == 0:
            num_itr_history.append(t+1)
            sup_err = get_sup_error_SGD(lam_min, lam_max, true_losses,
                                        model, test_data_loader, criterion, obj=obj)
            sup_err_history.append(sup_err)
            if (trace_frequency > 0) & ((t+1) % trace_frequency == 0):
                print(f"--------approximate solution path for # itr = {t+1} complete--------")
                print(f"# itr: {t+1}\t sup error: {sup_err}")

    return num_itr_history, sup_err_history
    
# return a list of loss computed on a specified grid over the solution path
def get_losses_SGD(model, lam_min, lam_max, num_grid, data_loader, loss_fn, obj=None):
    if obj is None:
        print("Please enter the objective: 'logit' or 'fairness'")
        return
    lambdas = np.linspace(lam_max, lam_min, num_grid)
    losses = []
    for lam in lambdas:
        if obj == "logit":
            losses.append(test_SGD(data_loader, model, loss_fn, lam))
        elif obj == "fairness":
            losses.append(fair_test_SGD(data_loader, model, loss_fn, lam))
            
    return losses
    
# return the absolute errors compared to the true loss accross the solution path  
def get_errs_SGD(model, lam_min, lam_max, true_loss_list, data_loader, loss_fn, obj=None):
    if obj is None:
        print("Please enter the objective: 'logit' or 'fairness'")
        return
    losses = get_losses_SGD(model, lam_min, lam_max, len(true_loss_list), data_loader, loss_fn, obj=obj)
    return losses - true_loss_list
    
# return the supremum absolute error compared to the true loss accross the solution path  
def get_sup_error_SGD(lam_min, lam_max, true_loss_list, model, data_loader, loss_fn, obj=None):
    if obj is None:
        print("Please enter the objective: 'logit' or 'fairness'")
        return
    errs = get_errs_SGD(model, lam_min, lam_max, true_loss_list, data_loader, loss_fn, obj=obj)
    return max(errs)