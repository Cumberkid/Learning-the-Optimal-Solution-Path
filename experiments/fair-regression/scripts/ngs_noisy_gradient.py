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

from lib.utils_data import Regression_Data
from lib.ngs.naive_grid_search import naive_grid_search
from lib.ngs.utils_ngs import get_sup_error

"""# Load data"""

# file path for Colab. May need to change this
X_df = pd.read_csv('../data/X_processed.csv')
y_df = pd.read_csv('../data/y_processed.csv')

X = np.array(X_df)
y = np.array(y_df).squeeze()

full_data = Regression_Data(X, y)
# stochastic gradient descent uses mini-batch
SGD_data_loader = DataLoader(full_data, batch_size=20, shuffle=True, )
# test data
test_data_loader = DataLoader(full_data, batch_size=len(full_data), shuffle=False, )

lam_max = 1
lam_min = 0
input_dim = X.shape[1]
criterion=torch.nn.BCELoss()


# Read the CSV file into a DataFrame
truth = pd.read_csv('../results/exact_soln_list.csv')

true_losses = truth['losses'].to_numpy()


"""# Noisy Gradient Oracle Diminishing LR

Use the previously tuned diminishing factor $\alpha = 8$ where lr = $\alpha/T$.
"""

lam_max = 1
lam_min = 0
alpha = 2**3
max_epochs = 5000
# a list of solution accuracy delta wish to be achieved
delta_list = 0.5 ** np.arange(3, 7, 0.125)

total_itr_list = []
sup_error_list = []
np.random.seed(8675309)
torch.manual_seed(8675309)
for delta in delta_list:
    # number of grids according to 1/sqrt(delta)
    num_grid = round(10 / np.sqrt(delta))

    start_time = time.time()
    total_itr, reg_params, intercepts, weights = naive_grid_search(lam_min=lam_min, lam_max=lam_max,
                                num_grid=num_grid, epochs=max_epochs, loss_fn=criterion,
                                trainDataLoader=SGD_data_loader, data_input_dim=input_dim, obj='fairness',
                                alpha=alpha, init_lr = 1, SGD=True, testDataLoader=test_data_loader,
                                true_loss_list=true_losses, stopping_criterion=delta)

    end_time = time.time()
    execution_time = end_time - start_time

    total_itr_list.append(total_itr)

    sup_error = get_sup_error(lam_min, lam_max, true_losses, intercepts,
                                  weights, reg_params, test_data_loader, criterion, obj='fairness')

    sup_error_list.append(sup_error)

    print(f"grid #: {num_grid}\t total iteration #: {total_itr}\t sup error: {sup_error}\t Execution time: {execution_time} seconds")

total_itr_list = np.array(total_itr_list)
sup_error_list = np.array(sup_error_list)

df = pd.DataFrame(np.column_stack((total_itr_list, sup_error_list)), columns=['num_itr', 'sup_err'])

# Save the DataFrame to a CSV file
df.to_csv('NGS_results_noisy.csv', index=False)
