import torch
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
