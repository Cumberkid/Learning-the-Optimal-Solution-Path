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


def C_bspline(numKnot, numPts):
    knots = np.linspace(0, 1, numKnot)
    spline_basis = SplineBasis(knots, 4)
    lams = np.linspace(0, 1, numPts)
    C = 0
    for lam in lams:
        v = spline_basis(lam)
        out = np.outer(v, v)
        eig = max(np.linalg.eigvals(out))
        C = max(C, eig)

    return C
