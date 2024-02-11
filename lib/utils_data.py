import numpy as np
import torch
from torch.utils.data import Dataset  #for creating the dataset

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

#A silly synthetic data example
def gen_data(n_obs_1, n_obs_2, n_var):
    # Draw X randomly
    data_1 = np.random.multivariate_normal(
             np.zeros(n_var),
             np.eye(n_var),
             n_obs_1
             )

    data_2 = np.random.multivariate_normal(
             np.ones(n_var),
             np.eye(n_var),
             n_obs_2
             )

    data = np.vstack((data_1, data_2))

    # Create corresponding labels (0 for the first distribution, 1 for the second)
    labels = np.hstack((np.zeros(n_obs_1), np.ones(n_obs_2)))

    # Shuffle the data and labels
    shuffled_indices = np.random.permutation(data.shape[0])
    data = data[shuffled_indices]
    labels = labels[shuffled_indices]
    return data.astype(np.float32), labels.astype(np.float32)
