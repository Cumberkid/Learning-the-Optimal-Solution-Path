import torch
from lib.ngs.log_reg_module import Logistic_Regression
from lib.ngs.reg_solver import test

"""Helper function that takes in a list of coarse grid models and returns the simulated losses and errors over $\lambda\in[0,1]$ compared to the exact solutions."""
# return the simulated losses accross the solution path
def get_losses(lam_min, lam_max, fine_delta_lam, intercepts, weights, reg_params, data_loader, loss_fn, device="cpu"):
    
    losses = []
    coarse_grid = 0
    weight = weights[coarse_grid].clone().detach()
    intercept = intercepts[coarse_grid]
    reg_param = reg_params[coarse_grid]
    model = Logistic_Regression(len(weight), 1, reg_param, weight, intercept).to(device)
    
    for i in range(round((lam_max - lam_min) / fine_delta_lam) + 1):
        lam = lam_max - i * fine_delta_lam
        if (coarse_grid + 1) < len(reg_params):
            if (reg_params[coarse_grid] - lam) > (lam - reg_params[coarse_grid + 1]):
                coarse_grid += 1
                model.reg_param = reg_params[coarse_grid]
                    
                with torch.no_grad():
                    model.linear.weight.copy_(weights[coarse_grid].clone().detach())
                    model.linear.bias.data.fill_(intercepts[coarse_grid])
        # approximate solution uses the linear weight of coarse grid model to test for regression parameter of the fine grid
        approx_loss = test(data_loader, model, loss_fn, lam, device)
        losses.append(approx_loss)
        # print(lam, coarse_grid)
    return losses

# return the absolute errors compared to the true loss accross the solution path
def get_errs(lam_min, lam_max, true_loss_list, intercepts, weights, reg_params, data_loader, loss_fn, device="cpu"):
    fine_delta_lam = (lam_max - lam_min)/(len(true_loss_list)-1)
    losses = get_losses(lam_min, lam_max, fine_delta_lam, intercepts, weights, reg_params, data_loader, loss_fn, device)
    errs = losses - true_loss_list
    return errs

# return the supremum absolute error compared to the true loss accross the solution path
def get_sup_error(lam_min, lam_max, true_loss_list, intercepts, weights, reg_params, data_loader, loss_fn, device="cpu"):
    errs = get_errs(lam_min, lam_max, true_loss_list, intercepts, weights, reg_params, data_loader, loss_fn, device)
    return max(errs)
