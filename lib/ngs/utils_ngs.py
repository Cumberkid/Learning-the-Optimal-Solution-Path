import torch
import numpy as np
from lib.ngs.log_reg_module import Logistic_Regression
from lib.ngs.solver import test
from scipy.interpolate import RegularGridInterpolator
from bisect import bisect_left


'''helper function that makes adjustment to hyperparameter position according to an already-found true list'''
def make_adjustment(new_params, true_loss_params, true_loss_list):
    idxes = []
    for param in new_params:
        true_loss_idx, true_loss_param = find_closest(param, true_loss_params)
        idxes.append(true_loss_idx)
    idxes = make_unique(idxes)
    # new_params_adjusted = true_loss_params(idxes)
    # selected_true_losses = true_loss_list(idxes)

    return true_loss_params[idxes], true_loss_list[idxes]

'''helper function that makes an array unique while maintaining the original order of appearance'''

def make_unique(array):
    seen = set()
    unique_array = []
    for item in array:
        if item not in seen:
            unique_array.append(item)
            seen.add(item)
    return unique_array

'''helper function that finds a closest hyperparameter in a list'''

def find_closest(lam, hyperparams):
    # Since the list is sorted in descending order, reverse it for bisect
    reversed_hyperparams = hyperparams[::-1]
    pos = bisect_left(reversed_hyperparams, lam)
    
    # Find the closest index in the original list
    if pos == 0:  # `a` is smaller or equal to the largest value
        closest_idx = len(hyperparams) - 1
    elif pos == len(hyperparams):  # `a` is larger or equal to the smallest value
        closest_idx = 0
    else:
        # Compare neighbors to find the closest in the original order
        left_idx = len(hyperparams) - pos
        right_idx = len(hyperparams) - pos - 1
        if abs(hyperparams[left_idx] - lam) <= abs(hyperparams[right_idx] - lam):
            closest_idx = left_idx
        else:
            closest_idx = right_idx
    
    # Retrieve corresponding value
    closest_hyperparam = hyperparams[closest_idx]
    return closest_idx, closest_hyperparam

"""Helper function that takes in a list of coarse grid models and returns the simulated losses and errors over $\lambda\in[0,1]$ compared to the exact solutions."""
# return the simulated losses accross the solution path
def get_losses(lam_min, lam_max, num_grid, intercepts, weights, hyper_params, data_loader, loss_fn, device="cpu"):
    
    losses = []
    coarse_grid = 0
    weight = weights[coarse_grid].clone().detach()
    intercept = intercepts[coarse_grid]
    hyper_param = hyper_params[coarse_grid]
    model = Logistic_Regression(len(weight), 1, hyper_param, weight, intercept).to(device)
    lambdas = np.linspace(lam_max, lam_min, num_grid)
    
    for lam in lambdas:
        if (coarse_grid + 1) < len(hyper_params):
            if (hyper_params[coarse_grid] - lam) > (lam - hyper_params[coarse_grid + 1]):
                coarse_grid += 1
                    
                with torch.no_grad():
                    model.linear.weight.copy_(weights[coarse_grid].clone().detach())
                    if model.linear.bias is not None:
                        model.linear.bias.data.fill_(intercepts[coarse_grid])
        # approximate solution uses the linear weight of coarse grid model to test for regression parameter of the fine grid
        approx_loss = test(data_loader, model, loss_fn, lam, device)
        losses.append(approx_loss)
        # print(lam, coarse_grid)
    return losses

# return the absolute errors compared to the true loss accross the solution path
def get_errs(lam_min, lam_max, true_loss_list, intercepts, weights, hyper_params, data_loader, loss_fn, device="cpu"):
    num_grid = len(true_loss_list)
    losses = get_losses(lam_min, lam_max, num_grid, intercepts, weights, hyper_params, data_loader, loss_fn, device)
    errs = np.array(losses) - np.array(true_loss_list)
    return errs

# return the supremum absolute error compared to the true loss accross the solution path
def get_sup_error(lam_min, lam_max, true_loss_list, intercepts, weights, hyper_params, data_loader, loss_fn, device="cpu"):
    errs = get_errs(lam_min, lam_max, true_loss_list, intercepts, weights, hyper_params, data_loader, loss_fn, device)
    return np.max(errs)

def get_losses_linear(interp_points, intercepts, weights, hyper_params, data_loader, loss_fn, device="cpu"):
    
    losses = []
    weight = weights[0].clone().detach()
    intercept = intercepts[0]
    hyper_param = hyper_params[0]
    model = Logistic_Regression(len(weight), 1, hyper_param, weight, intercepts[0]).to(device)
    lambdas = interp_points
    
    weight_interp = RegularGridInterpolator((hyper_params,), weights)
    intercept_interp = RegularGridInterpolator((hyper_params,), intercepts)

    lambdas = interp_points

    for lam in lambdas:
        weight = torch.tensor(weight_interp([lam]))
        intercept = intercept_interp([lam])[0]
        with torch.no_grad():
            model.linear.weight.copy_(weight)
            if model.linear.bias is not None:
                model.linear.bias.data.fill_(intercept)
        # approximate solution uses the linear weight of coarse grid model to test for regression parameter of the fine grid
        approx_loss = test(data_loader, model, loss_fn, lam, device)
        losses.append(approx_loss)
        # print(lam, coarse_grid)
    return losses

# return the absolute errors compared to the true loss across the solution path corresponding to a list of params
def get_errs_linear(true_loss_params, true_loss_list, intercepts, weights, hyper_params, data_loader, loss_fn, device="cpu"):
    # num_grid = len(true_loss_list)
    losses = get_losses_linear(true_loss_params, intercepts, weights, hyper_params, data_loader, loss_fn, device)
    errs = np.array(losses) - np.array(true_loss_list)
    return errs

# return the supremum absolute error compared to the true loss across the solution path corresponding to a list of params
def get_sup_error_linear(true_loss_params, true_loss_list, intercepts, weights, hyper_params, data_loader, loss_fn, device="cpu"):
    errs = get_errs_linear(true_loss_params, true_loss_list, intercepts, weights, hyper_params, data_loader, loss_fn, device)
    return np.max(errs)
