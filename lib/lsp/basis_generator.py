import torch
import math
import numpy as np
from scipy.interpolate import BSpline
from scipy.special import legendre, eval_laguerre, eval_chebyu

# monomials
def monomials(lam, basis_dim, device='cpu'):
    vec = torch.tensor([lam**i for i in range(basis_dim)], dtype=torch.float32)
    return bec.to(device)
    
# scaled and shifted Legendre polynomials
def scaled_shifted_legendre(lam, basis_dim, device='cpu'):
    # Transform the lam to [-1, 1] interval
    lam_transformed = 2 * lam - 1
    vec = torch.tensor([math.sqrt(2*i+1) * legendre(i)(lam_transformed) for i in range(basis_dim)], dtype=torch.float32)
    return vec.to(device)

# dampened Laguerre polynomials
def dampen_laguerre(lam, basis_dim, device='cpu'):
    vec = torch.tensor([(np.sqrt(10) * np.exp(-0.45 * lam) * eval_laguerre(i, lam)) for i in range(basis_dim)], dtype=torch.float32)
    return vec.to(device)

# chebyshev polynomial of the second kind
def chebyshev_second_kind(lam, basis_dim, device='cpu'):
    vec = torch.tensor([eval_chebyu(i, lam) for i in range(basis_dim)], dtype=torch.float32)
    return vec.to(device)
    
# cubic bspline basis
class SplineBasis:
    def __init__(self, knots, order):
      self.order = order
      #pad the knots
      taus = np.hstack((knots[0] * np.ones(order - 1), knots, knots[-1] * np.ones(order - 1)))
      #iterate through and create separate spline objects
      self.basis = []
      numBasis = len(knots) + order - 2
      for i in range(numBasis):
        ei = np.zeros(numBasis)
        ei[i] = 1.
        spl = BSpline(taus, ei, order - 1)
        self.basis.append(spl)

    def __call__(self, pt):
      return( [float(b(pt)) for b in self.basis] )

    def numBasisFns(self):
      return(len(self.basis))

# cubic bspline takes basis_dim at least 6
def cubic_bspline(lam, basis_dim, device='cpu'):
    order = 4 # 4 for cubic
    numKnot = basis_dim + 2 - order
    knots = np.linspace(0, 1, numKnot)
    spline_basis = SplineBasis(knots, order)
    vec = torch.tensor(spline_basis(lam), dtype=torch.float32)
    return vec.to(device)
