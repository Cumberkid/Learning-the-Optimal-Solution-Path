import numpy as np
import torch
from scipy.special import roots_jacobi, beta as beta_func
from scipy.stats import semicircular, uniform, expon, beta
from numpy.polynomial.legendre import leggauss

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, weights, lam, learning_rate, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        y_pred = sigmoid(np.dot(X, weights))
        gradient = (1-lam) * np.dot(X.T, y_pred - y) / m + lam * weights
        weights -= learning_rate * gradient
    return weights

def logit_by_hand(weight, intercept, lam, X_set, y_set):
    # weight = torch.zeros(input_dim, 1)
    pred = torch.sigmoid(torch.mm(X_set, weight) + intercept)
    # print(pred)
    criterion = torch.nn.BCELoss()
    soln = (1 - lam) * criterion(torch.squeeze(pred), y_set)
    soln += lam * 0.5 * (torch.squeeze(weight).norm(p=2)**2 + torch.squeeze(intercept)**2)

    return soln, pred

"""Quadrature functions"""

#converts a random sample from [-1, 1] to a general interval [a, b] 
def u_to_lam(u, a, b):
    return (u + 1) * 0.5 * (b - a) + a

def gauss_legendre_integral(func, a, b, num_quadpts):
    # Get the points and weights for Gauss-Legendre quadrature on [-1, 1]
    points, weights = leggauss(num_quadpts)

    out = 0.0
    for i in range(num_quadpts):
        lam = u_to_lam(points[i], a, b)
        out += 0.5 * (b - a) * weights[i] * func(lam)
    
    return out

def gauss_legendre_integral_2d(func, lam_min, lam_max, num_quadpts):
    points, weights = leggauss(num_quadpts)
    
    out = 0.0
    for i in range(num_quadpts):
        for j in range(num_quadpts):
            lam1 = u_to_lam(points[i], lam_min[0], lam_max[0])
            lam2 = u_to_lam(points[j], lam_min[1], lam_max[1])
            out += 0.5 * (lam_max[0] - lam_min[0]) * 0.5 * (lam_max[1] - lam_min[1]) * weights[i] * weights[j] * func([lam1, lam2])
    
    return out

def monte_carlo(func, lam_min, lam_max, num_pts, distribution, random_state=88):
    out = 0.0
    np.random.seed(random_state)
    for j in range(num_pts):
        hyper_params = np.zeros(len(lam_max))
        if distribution == 'uniform':
            samples = uniform.rvs(loc=0, scale=1, size=len(lam_min))
            hyper_params = samples * (lam_max - lam_min) + lam_min
        # elif distribution == 'exponential':
        #     hyper_params[i] = expon.rvs(scale=(lam_max[i] - lam_min[i])/2)
        #     while hyper_params[i] > (lam_max[i] - lam_min[i]):
        #         hyper_params[i]/=2
        # elif distribution == 'semicircle':
        #     hyper_params[i] = semicircular.rvs(loc=lam_min[i], scale=lam_max[i]-lam_min[i])
        # elif distribution == 'beta':
        #     # Sample from the standard Beta distribution on [0, 1] with flipped alpha, beta
        #     # Then transform to the interval [a, b]
        #     hyper_params[i] = beta.rvs(alpha_beta[1] + 1, alpha_beta[0] + 1, loc=lam_min[i], scale=lam_max[i]-lam_min[i])
        out += func(hyper_params)
        # print(hyper_params)
    
    return out/num_pts

# def gauss_jacobi_expectation(func, alpha, beta, a, b, n_points):
#     # Compute Gauss-Jacobi points and weights for interval [-1, 1] with parameters alpha, beta
#     points, weights = roots_jacobi(n_points, alpha, beta)
    
#     # Transform the points to the interval [a, b]
#     transformed_points = u_to_lam(points, a, b)
#     adjusted_weights = 0.5 * (b - a) * weights
    
#     # Beta distribution normalization factor over [a, b]
#     normalization = 1 / (beta_func(beta+1, alpha+1) * (b - a)**(alpha+beta+1))
    
#     # Compute the expectation as a weighted sum of function values at transformed points
#     expectation = np.sum(adjusted_weights * func(transformed_points) *
#                          normalization * ((transformed_points - a)**beta) *
#                          ((b - transformed_points)**alpha))
    
#     return expectation