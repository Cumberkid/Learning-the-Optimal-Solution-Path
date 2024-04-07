import torch
import math
from scipy.special import legendre

# scaled and shifted legendre polynomials
def scaled_shifted_legendre(lam, basis_dim, device='cpu'):
    # Transform the lam to [-1, 1] interval
    lam_transformed = 2 * lam - 1
    vec = torch.tensor([math.sqrt(2*i+1) * legendre(i)(lam_transformed) for i in range(basis_dim)])
    return vec.to(device)


