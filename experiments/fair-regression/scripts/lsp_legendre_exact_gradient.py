# -*- coding: utf-8 -*-
"""07 Learn Solution Path (LSP): Legendre.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/Cumberkid/Learning-the-Optimal-Solution-Path/blob/main/experiments/fair-regression/notebooks/07%20Learn%20Solution%20Path%20(LSP)%3A%20Legendre.ipynb

Runs learn_solution_path with exact gradient oracle and diminishing learning rate.

# Import necessary libraries
"""

import numpy as np
import torch
from torch.utils.data import DataLoader  #for creating the dataset


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

import pandas as pd

"""## Import our own modules"""

import lib
# importlib.reload(lib)

from lib.utils_data import Regression_Data
from lib.lsp.basis_generator import phi_lam_legendre
from lib.lsp.basis_tf_module import Basis_TF_SGD
from lib.lsp.learn_solution_path import learn_solution_path

"""#Preliminaries
Recall that our method runs SGD over random $\tilde λ$'s with a linear basis $\Phi(\tilde \lambda)$ of our choice. We want to approximate $\theta$ with $\Phi(\lambda)\beta$, so the objective function is $\min_\beta h(\Phi(\tilde\lambda)\beta, \tilde\lambda) = (1-\tilde\lambda) BCE(X_\text{pass}\Phi(\tilde\lambda)\beta,\ y_\text{pass}) + \tilde\lambda BCE(X_\text{fail}\Phi(\tilde\lambda)\beta,\ y_\text{fail})$. For each batch of training data set, we randomize $\tilde\lambda$. If batch size = 1, then this is equivalent to a standard SGD.

## Load data
"""

# file path for Colab. May need to change this
X_df = pd.read_csv('../data/X_processed.csv')
y_df = pd.read_csv('../data/y_processed.csv')

X = np.array(X_df)
y = np.array(y_df).squeeze()

full_data = Regression_Data(X, y)
# full gradient descent uses all data points
GD_data_loader = DataLoader(full_data, batch_size=len(full_data), shuffle=True, )
# test data
test_data_loader = DataLoader(full_data, batch_size=len(full_data), shuffle=False, )

lam_max = 1
lam_min = 0
input_dim = X.shape[1]
criterion=torch.nn.BCELoss()

"""## Choose basis functions

We use Legendre polynomials with degree $\leq n$ as the basis vectors for $\Phi(\lambda)$.
"""

phi_lam = phi_lam_legendre

criterion = torch.nn.BCELoss()
input_dim = X.shape[1]

"""# Exact Gradient Oracle Diminishing LR"""
# Read the CSV file into a DataFrame
truth = pd.read_csv('../results/exact_soln_list.csv')

true_losses = truth['losses'].to_numpy()

"""We use diminishing learning rate for better demonstrate convergence. If we use a constant learning rate, the solution path error will eventually do a random walk after descending to a certain threshold value.

We will see this random walk in a plot later.
"""

epochs = 20000

"""## num basis func = 3. Change basis_dim if a different number of functions is used."""

basis_dim = 3
lr = 0.5
gamma=0.97        # diminishing factor
dim_step=100        # diminishing frequency

np.random.seed(8675309)
torch.manual_seed(8675309)

num_itr_history, sup_err_history, model = learn_solution_path(input_dim, basis_dim, phi_lam, epochs,
                                                                          GD_data_loader, test_data_loader, criterion,
                                                                          lam_min, lam_max, true_losses,
                                                                          lr=lr, diminish=True, gamma=gamma, dim_step=dim_step,
                                                                          obj='fairness',
                                                                          record_frequency=50, trace_frequency=100)
num_itr_history = np.array(num_itr_history)
sup_err_history = np.array(sup_err_history)

file_path = 'SGD_results_exact_diminish.csv'

# Read the CSV file into a DataFrame if already exist
# df = pd.read_csv(file_path)
# df[f'sup_err_{basis_dim}'] = sup_err_history

# Change the column name 'sup_err_x' according to basis_dim
df = pd.DataFrame(np.column_stack((num_itr_history, sup_err_history)), columns=['num_itr', 'sup_err_3'])

# Save the DataFrame to a CSV file
df.to_csv(file_path, index=False)

