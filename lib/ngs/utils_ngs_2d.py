import torch
import numpy as np
from lib.ngs.portfolio_alloc_module import Portfolio_Allocation
from lib.ngs.solver import test

"""Helper function that takes in a list of coarse grid models and returns the simulated losses and errors over $\lambda\in[0,1]$ compared to the exact solutions."""

# return the simulated losses accross the solution path
def get_losses(fix_lam, lam_min, lam_max, num_grid, intercepts, weights, hyper_params_list, data_loader, loss_fn, device="cpu"):

    losses = []
    coarse_grid = 0
    weight = weights[coarse_grid].clone().detach()
    intercept = intercepts[coarse_grid]
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
                    if model.bias is not None:
                        model.linear.bias.data.fill_(intercepts[coarse_grid])
        # approximate solution uses the linear weight of coarse grid model to test for regression parameter of the fine grid
        approx_loss = test(data_loader, model, loss_fn, [fix_lam, lam], device)
        losses.append(approx_loss)
        # print(lam, coarse_grid)
    return losses

# return the absolute errors compared to the true loss accross the solution path
def get_errs(lam_min, lam_max, true_loss_list, intercepts, weights, hyper_params, data_loader, loss_fn, device="cpu"):
    fine_delta_lam = (lam_max - lam_min)/(len(true_loss_list)-1)
    losses = get_losses(lam_min, lam_max, fine_delta_lam, intercepts, weights, hyper_params, data_loader, loss_fn, device)
    errs = losses - true_loss_list
    return errs

# return the supremum absolute error compared to the true loss accross the solution path
def get_sup_error(lam_min, lam_max, true_loss_list, intercepts, weights, hyper_params, data_loader, loss_fn, device="cpu"):
    errs = get_errs(lam_min, lam_max, true_loss_list, intercepts, weights, hyper_params, data_loader, loss_fn, device)
    return max(errs)

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

