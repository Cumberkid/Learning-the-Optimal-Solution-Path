import torch

# monomials
def generate_monomials(lam, basis_dim, device='cpu'):
    vec = torch.tensor([lam**i for i in range(basis_dim)])
    return bec.to(device)
