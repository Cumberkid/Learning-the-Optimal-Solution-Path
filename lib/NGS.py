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
from torch.optim.lr_scheduler import StepLR
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

# this initializes with random weights. Need to either set a seed or force initialization somewhere for reproducibility.
# automatically fits an intercept. To turn off intercept, set bias=False in nn.Linear()
class Logistic_Regression(nn.Module):
    def __init__(self, input_dim, output_dim, reg_param, init_weight, init_intercept):
        super(Logistic_Regression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.actv = nn.Sigmoid()
        self.reg_param = reg_param
        
        # initialize for better performance
        with torch.no_grad():
          self.linear.weight.copy_(init_weight)
          self.linear.bias.data.fill_(init_intercept)
          
    def forward(self, x):
        return self.actv(self.linear(x))
        
    def ridge_term(self):
        return self.linear.weight.norm(p=2)**2 + self.linear.bias.norm(p=2)**2

"""The "train" function executes optimization on the input dataset w.r.t. the input loss function with the input optimizer on the ridge-regularized regression objective $h(\theta, \lambda) = (1-\lambda)BCE(X\theta, y) + \frac{\lambda}{2}\|\theta\|^2$. We will use the pytorch built-in SGD optimizer later, but note that this optimizer is actually just a deterministic gradient descent program.

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

"""The "fair_train" function is similar to the "train" function except that its objective function aims to treat two groups with fairness, i.e. $h(\theta, \lambda) = (1-\lambda)*loss(X_{\text{group A}}\theta, y_{\text{group A}}) + \lambda*loss(X_{\text{group B}}\theta, y_{\text{group B}})$.
"""

# trace_frequency is measured in number of batches. -1 means don't print
def fair_train(dataloader, model, loss_fn, optimizer, trace_frequency = -1):
    # size = len(dataloader.dataset)
    model.train()
    # here, the "batch" notion takes care of randomization
    for batch, (X_train, y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)
        # print(batch, len(X_train))
        X_major = X_train[y_train == 1]
        y_major = torch.ones(len(X_major)).to(device)
        X_minor = X_train[y_train == 0]
        y_minor = torch.zeros(len(X_minor)).to(device)
        
        # Compute predicted y_hat
        pred_major = model(X_major)
        pred_minor = model(X_minor)
        
        # Fair loss function
        loss = (1 - model.reg_param) * loss_fn(pred_major.view(-1, 1), y_major.view(-1, 1))
        loss += model.reg_param * loss_fn(pred_minor.view(-1, 1), y_minor.view(-1, 1))
                
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # if (trace_frequency > 0) & (batch % trace_frequency == 0):
        #     loss, current = loss.item(), (batch + 1) * len(X_train)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

"""The "fair_test" function defined here is our objective function $h(\theta, \lambda) = (1-\lambda)*loss(X_{\text{group A}}\theta, y_{\text{group A}}) + \lambda*loss(X_{\text{group B}}\theta, y_{\text{group B}})$. The linear weight from the above trained model is our $\theta$."""

# Test function
def fair_test(dataloader, model, loss_fn, lam):
    model.eval() # important
    with torch.no_grad():  # makes sure we don't corrupt gradients and is faster
        for batch, (X_test, y_test) in enumerate(dataloader):
            X_test, y_test = X_test.to(device), y_test.to(device)
            
            X_major = X_test[y_test == 1]
            y_major = torch.ones(len(X_major)).to(device)
            X_minor = X_test[y_test == 0]
            y_minor = torch.zeros(len(X_minor)).to(device)
            
            # Compute predicted y_hat
            pred_major = model(X_major)
            pred_minor = model(X_minor)
            
            # With regularization
            oos = (1 - lam) * loss_fn(pred_major.view(-1, 1), y_major.view(-1, 1)) 
            oos += lam * loss_fn(pred_minor.view(-1, 1), y_minor.view(-1, 1))
                    
    return oos.item()

"""# Naive Grid Search"""

# running gradient descent with fixed learning rate on a single grid point, i.e. for one specified lambda
def GD_on_a_grid(lam, lam_max, epochs, loss_fn, model, optimizer, trainDataLoader, data_input_dim,
                 obj=None, alpha=1, init_lr=0.1, diminish=False, gamma=0.1, dim_step=30, SGD=False, 
                 testDataLoader=None, true_loss_list=None, fine_delta_lam=None, stopping_criterion=None):
                     
    if true_loss_list is not None:
        # true loss
        i = round((lam_max - lam) / fine_delta_lam)
        if i >= len(true_loss_list):
            i = len(true_loss_list) - 1
        true_loss = true_loss_list[i]
        lam = lam_max - i * fine_delta_lam
        # print(f"nearest i = {i}\t lam = {lam}")
        
    model.reg_param = lam
    if diminish:
        # Define the learning rate scheduler
        scheduler = StepLR(optimizer, step_size=dim_step, gamma=gamma)  # Decrease LR by a factor of gamma every dim_step epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = init_lr  
        
    early_stop = False
    itr = 0
    for t in range(epochs):
        if SGD:
            # shrink learning rate:
            lr = min([init_lr, alpha/(t+1)])
            optimizer.zero_grad()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
        if obj == "logit":
            train(trainDataLoader, model, loss_fn, optimizer)
        elif obj == "fairness":
            fair_train(trainDataLoader, model, loss_fn, optimizer)
            
        if true_loss_list is not None:
            if (t+1) % 10 == 0:
                # do an accuracy check
                if obj == "logit":
                    approx_loss = test(testDataLoader, model, loss_fn, lam)
                elif obj == "fairness":
                    approx_loss = fair_test(testDataLoader, model, loss_fn, lam)
                    
                error = approx_loss - true_loss
                # print(lr, error, true_loss)
                # stopping criterion
                if error <= stopping_criterion:
                    itr += (t+1)
                    early_stop = True
                    break  # Early stop
                
        if diminish:
            # Update the learning rate
            scheduler.step()
            
    if not early_stop:
        itr += epochs
        
    return itr

"""Naive Grid Search starts from $\lambda = 1$ and decreases $\lambda$ by $\Delta\lambda = \frac{\lambda_\text{max} - \lambda_\text{min}}{\text{# of grid}}$. The model trained on each grid point $(\lambda - \Delta\lambda)$ initializes weight with the linear weight of the model trained on the previous grid point $\lambda$."""

# do the whole naive grid search over a list of uniformly spaced lambda's
# from lam_min to lam_max
# returns a list of trained models
def naive_grid_search(lam_min, lam_max, num_grid, epochs, loss_fn, trainDataLoader,
                      data_input_dim, obj=None, lr=1e-3, alpha=1, init_lr=1, 
                      diminish=False, gamma=0.1, dim_step=30, SGD=False,
                      testDataLoader=None, true_loss_list=None, stopping_criterion=None):
    if obj is None:
        print("Please enter the objective: 'logit' or 'fairness'")
        return
    
    fine_delta_lam = None
    if true_loss_list is not None:
        fine_delta_lam = (lam_max - lam_min)/(len(true_loss_list) - 1)
        
    reg_params = []
    weights = []
    intercepts = []
    total_itr = 0
    # create a list of lambda's
    lambdas = np.linspace(lam_max, lam_min, num_grid)
    
    # first weight is initialized at 0
    weight = torch.zeros(data_input_dim)
    intercept = 0
    model = Logistic_Regression(data_input_dim, 1, lam_max, weight, intercept).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    
    for lam in lambdas:
        # print(f"Running model on lambda = {lam}")
        itr = GD_on_a_grid(lam, lam_max, epochs, loss_fn, model, optimizer,
                                  trainDataLoader=trainDataLoader,
                                  data_input_dim=data_input_dim,
                                  obj=obj, alpha=alpha, 
                                  init_lr=init_lr, diminish=diminish, 
                                  gamma=gamma, dim_step=dim_step, SGD=SGD, 
                                  testDataLoader=testDataLoader,
                                  true_loss_list=true_loss_list,
                                  fine_delta_lam=fine_delta_lam,
                                  stopping_criterion=stopping_criterion)
        weights.append(model.linear.weight.clone().data.cpu().numpy()[0])
        intercepts.append(model.linear.bias.clone().data.cpu().numpy()[0])
        # print(model.linear.weight)
        reg_params.append(model.reg_param)
        total_itr += itr
        # print(total_itr)
        
    return total_itr, reg_params, intercepts, weights

"""Helper function that takes in a list of coarse grid models and returns the simulated losses and errors over $\lambda\in[0,1]$ compared to the exact solutions."""
# return the simulated losses accross the solution path
def get_losses(lam_min, lam_max, fine_delta_lam, intercepts, weights, reg_params, data_loader, loss_fn, obj=None):
    
    losses = []
    coarse_grid = 0
    weight = torch.tensor(weights[coarse_grid])
    intercept = intercepts[coarse_grid]
    reg_param = reg_params[coarse_grid]
    model = Logistic_Regression(len(weight), 1, reg_param, weight, intercept).to(device)
    if obj is None:
        print("Please enter the objective: 'logit' or 'fairness'")
        return
    
    for i in range(round((lam_max - lam_min) / fine_delta_lam) + 1):
        lam = lam_max - i * fine_delta_lam
        if (coarse_grid + 1) < len(reg_params):
            if (reg_params[coarse_grid] - lam) > (lam - reg_params[coarse_grid + 1]):
                coarse_grid += 1
                model.reg_param = reg_params[coarse_grid]
                    
                with torch.no_grad():
                    model.linear.weight.copy_(torch.tensor(weights[coarse_grid]))
                    model.linear.bias.data.fill_(intercepts[coarse_grid])
        # approximate solution uses the linear weight of coarse grid model to test for regression parameter of the fine grid
        if obj == "logit":
            approx_loss = test(data_loader, model, loss_fn, lam)
        elif obj == "fairness":
            approx_loss = fair_test(data_loader, model, loss_fn, lam)
        losses.append(approx_loss)
        # print(lam, coarse_grid)
    return losses

# return the absolute errors compared to the true loss accross the solution path
def get_errs(lam_min, lam_max, true_loss_list, intercepts, weights, reg_params, data_loader, loss_fn, obj=None):
    fine_delta_lam = (lam_max - lam_min)/(len(true_loss_list)-1)
    if obj is None:
        print("Please enter the objective: 'logit' or 'fairness'")
        return
    losses = get_losses(lam_min, lam_max, fine_delta_lam, intercepts, weights, reg_params, data_loader, loss_fn, obj=obj)
    errs = losses - true_loss_list
    return errs

# return the supremum absolute error compared to the true loss accross the solution path
def get_sup_error(lam_min, lam_max, true_loss_list, intercepts, weights, reg_params, data_loader, loss_fn, obj=None):
    if obj is None:
        print("Please enter the objective: 'logit' or 'fairness'")
        return
    errs = get_errs(lam_min, lam_max, true_loss_list, intercepts, weights, reg_params, data_loader, loss_fn, obj=obj)
    return max(errs)