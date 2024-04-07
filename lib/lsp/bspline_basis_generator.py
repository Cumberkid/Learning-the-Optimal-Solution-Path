import torch
import math
import numpy as np
from scipy.interpolate import BSpline

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
def bspline(lam, basis_dim, device='cpu'):
    order = 4 # 4 for cubic
    numKnot = basis_dim + 2 - order
    knots = np.linspace(0, 1, numKnot)
    spline_basis = SplineBasis(knots, order)
    vec = torch.tensor(spline_basis(lam))
    return vec.to(device)
