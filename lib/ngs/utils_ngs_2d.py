import torch
import numpy as np
from lib.ngs.portfolio_alloc_module import Portfolio_Allocation
from lib.ngs.solver import test
from scipy.interpolate import RegularGridInterpolator
from bisect import bisect_left
from lib.ngs.utils_ngs import find_closest, make_unique

'''helper function that makes adjustment to hyperparameter position according to an already-found true list'''
def make_adjustment_2d(new_params, true_loss_params, true_loss_list):
    new_params_1d = new_params[:, 0, 0]
    idxes = []
    for param in new_params_1d:
        true_loss_idx, true_loss_param = find_closest(param, true_loss_params[:, 0, 0])
        idxes.append(true_loss_idx)
    idxes = make_unique(idxes)
    # new_params_adjusted = true_loss_params(idxes)
    # selected_true_losses = true_loss_list(idxes)

    return true_loss_params[idxes], true_loss_list[idxes]


'''Helper funciton that gets the first and last column of an inhomogenous array'''
def get_first_last(array):
    # Extract the first elements
    first_column = [row[0] for row in array]
    # Extract the last elements
    last_column = [row[-1] for row in array]
    # Stack the first and last columns, preserving sub-list structure
    result = np.stack((first_column, last_column), axis=1)

    return result
"""Helper function that takes in a list of coarse grid models and returns the simulated losses and errors over $\lambda\in[0,1]$ compared to the exact solutions."""

# return the simulated losses accross the solution path
def get_losses(fix_lam, lam_min, lam_max, num_grid, intercepts, weights, hyper_params_list, data_loader, loss_fn, device="cpu"):

    losses = []
    coarse_grid = 0
    weight = weights[coarse_grid].clone().detach()
    hyper_params = [fix_lam, hyper_params_list[coarse_grid]]
    model = Portfolio_Allocation(len(weight), 1, hyper_params, weight).to(device)
    lambdas = np.linspace(lam_max, lam_min, num_grid)
    
    for lam in lambdas:
        if (coarse_grid + 1) < len(hyper_params_list):
            if (hyper_params_list[coarse_grid] - lam) > (lam - hyper_params_list[coarse_grid + 1]):
                coarse_grid += 1
                # model.hyper_params = [fix_lam, hyper_params_list[coarse_grid]]
                    
                with torch.no_grad():
                    model.linear.weight.copy_(weights[coarse_grid].clone().detach())
                    if model.linear.bias is not None:
                        model.linear.bias.data.fill_(intercepts[coarse_grid])
        # approximate solution uses the linear weight of coarse grid model to test for regression parameter of the fine grid
        approx_loss = test(data_loader, model, loss_fn, [fix_lam, lam], device)
        losses.append(approx_loss)
        # print(lam, coarse_grid)
    return losses


# returns a 2-d numpy array
def get_losses_2d(lam_min_2d, lam_max_2d, num_grid_2d, intercepts_2d, weights_2d, hyper_params_2d, data_loader, loss_fn, device="cpu"):
    losses_2d = []
    coarse_grid = 0
    lambdas = np.linspace(lam_max_2d[0], lam_min_2d[0], num_grid_2d[0])

    for fix_lam in lambdas:
        if (coarse_grid + 1) < len(hyper_params_2d):
            if (hyper_params_2d[coarse_grid][0][0] - fix_lam) > (fix_lam - hyper_params_2d[coarse_grid + 1][0][0]):
                coarse_grid += 1
        # print(fix_lam, hyper_params_2d[coarse_grid][0][0])
        losses = get_losses(fix_lam, lam_min_2d[1], lam_max_2d[1], num_grid_2d[1], intercepts_2d[coarse_grid], 
                            weights_2d[coarse_grid], hyper_params_2d[coarse_grid][:, 1], data_loader, loss_fn, device)
        losses_2d.append(losses)

    return losses_2d

# true_loss_list_2d must be numpy array input
# returns a 2-d numpy array
def get_errs_2d(lam_min_2d, lam_max_2d, true_loss_list_2d, intercepts_2d, weights_2d, hyper_params_2d, data_loader, loss_fn, device="cpu"):
    num_grid_2d = [len(true_loss_list_2d), len(true_loss_list_2d[0])]
    losses_2d = get_losses_2d(lam_min_2d, lam_max_2d, num_grid_2d, intercepts_2d, weights_2d, hyper_params_2d, data_loader, loss_fn, device)
    errs_2d = losses_2d - true_loss_list_2d
    return errs_2d

# eturn the supremum absolute error compared to the true loss accross the 2-d solution path
def get_sup_error_2d(lam_min_2d, lam_max_2d, true_loss_list_2d, intercepts_2d, weights_2d, hyper_params_2d, data_loader, loss_fn, device="cpu"):
    errs_2d = get_errs_2d(lam_min_2d, lam_max_2d, true_loss_list_2d, intercepts_2d, weights_2d, hyper_params_2d, data_loader, loss_fn, device)
    return np.max(np.array(errs_2d))


'''Use linear interpolation between grids'''

def get_solns_linear(interp_points, intercepts, weights, hyper_params):
    
    running_lams = np.array(hyper_params)[:, 1]

    weight_interp = RegularGridInterpolator((running_lams, ), weights)
    intercept_interp = RegularGridInterpolator((running_lams, ), intercepts)

    lambdas = np.array(interp_points)[:, 1]

    interp_weights = weight_interp(lambdas)
    interp_intercepts = intercept_interp(lambdas)

    return interp_weights, interp_intercepts

# returns a 1-d numpy array
def get_losses_linear(interp_points, intercepts, weights, hyper_params, data_loader, loss_fn, device="cpu"):
    losses = []
    interp_weights, interp_intercepts = get_solns_linear(interp_points, intercepts, weights, hyper_params)
    weight = torch.tensor(interp_weights[0])
    intercept = interp_intercepts[0]
    hyper_params = interp_points[0]
    model = Portfolio_Allocation(len(weight), 1, hyper_params, weight).to(device)

    losses = []
    for weight, intercept, hyper_params in zip(interp_weights, interp_intercepts, interp_points):
        with torch.no_grad():
            model.linear.weight.copy_(torch.tensor(weight))
            if model.linear.bias is not None:
                model.linear.bias.data.fill_(intercept)
        # approximate solution uses the linear weight of coarse grid model to test for regression parameter of the fine grid
        loss = test(data_loader, model, loss_fn, hyper_params, device)
        losses.append(loss)

    return losses

# true_loss_list must be numpy array input
# returns a 1-d numpy array
def get_errs_linear(interp_points, true_loss_list, intercepts, weights, hyper_params, data_loader, loss_fn, device="cpu"):
    losses = get_losses_linear(interp_points, intercepts, weights, hyper_params, data_loader, loss_fn, device)
    errs = np.array(losses) - np.array(true_loss_list)
    return errs


# returns a 2-d numpy array
def get_solns_linear_2d(interp_points_2d, intercepts_2d, weights_2d, hyper_params_2d):
    interp_weights_2d = []
    interp_intercepts_2d = []
    lambdas = get_first_last(hyper_params_2d)[:, 0, 0]
    # print(lambdas)

    for interp_points in interp_points_2d:
        idx_left = bisect_left([-lam for lam in lambdas], -interp_points[0][0])
        weights_left, intercepts_left = get_solns_linear(interp_points, intercepts_2d[idx_left], 
                                                        weights_2d[idx_left], hyper_params_2d[idx_left])
        if idx_left+1 < len(hyper_params_2d):
            weights_right, intercepts_right = get_solns_linear(interp_points, intercepts_2d[idx_left+1], 
                                                            weights_2d[idx_left+1], hyper_params_2d[idx_left+1])
            prop_left = (lambdas[idx_left] - interp_points[0][0]) / (lambdas[idx_left] - lambdas[idx_left+1])
            
        else:
            weights_right, intercepts_right = weights_left, intercepts_left
            prop_left = 1
        
        interp_weights = prop_left * weights_left + (1 - prop_left) * weights_right
        interp_intercepts = prop_left * intercepts_left + (1 - prop_left) * intercepts_right
        interp_weights_2d.append(interp_weights)
        interp_intercepts_2d.append(interp_intercepts)

    return interp_weights_2d, interp_intercepts_2d

# returns a 2-d numpy array
def get_losses_linear_2d(interp_points_2d, intercepts_2d, weights_2d, hyper_params_2d, data_loader, loss_fn, device="cpu"):
    losses_2d = []
    interp_weights_2d, interp_intercepts_2d = get_solns_linear_2d(interp_points_2d, intercepts_2d, weights_2d, hyper_params_2d)
    weight = torch.tensor(interp_weights_2d[0][0])
    intercept = interp_intercepts_2d[0][0]
    hyper_params = interp_points_2d[0][0]
    model = Portfolio_Allocation(len(weight), 1, hyper_params, weight).to(device)

    for interp_weights, interp_intercepts, interp_points in zip(interp_weights_2d, interp_intercepts_2d, interp_points_2d):
        losses = []
        for weight, intercept, hyper_params in zip(interp_weights, interp_intercepts, interp_points):
            with torch.no_grad():
                model.linear.weight.copy_(torch.tensor(weight))
                if model.linear.bias is not None:
                    model.linear.bias.data.fill_(intercept)
            # approximate solution uses the linear weight of coarse grid model to test for regression parameter of the fine grid
            loss = test(data_loader, model, loss_fn, hyper_params, device)
            losses.append(loss)
        losses_2d.append(losses)

    return losses_2d

# true_loss_list_2d must be numpy array input
# returns a 2-d numpy array
def get_errs_linear_2d(true_params_2d, true_loss_list_2d, intercepts_2d, weights_2d, hyper_params_2d, data_loader, loss_fn, device="cpu"):
    losses_2d = get_losses_linear_2d(true_params_2d, intercepts_2d, weights_2d, hyper_params_2d, data_loader, loss_fn, device)
    errs_2d = np.array(losses_2d) - np.array(true_loss_list_2d)
    return errs_2d

# eturn the supremum absolute error compared to the true loss accross the 2-d solution path
def get_sup_error_linear_2d(true_params_2d, true_loss_list_2d, intercepts_2d, weights_2d, hyper_params_2d, data_loader, loss_fn, device="cpu"):
    errs_2d = get_errs_linear_2d(true_params_2d, true_loss_list_2d, intercepts_2d, weights_2d, hyper_params_2d, data_loader, loss_fn, device)
    return np.max(errs_2d)