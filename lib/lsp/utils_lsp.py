import numpy as np
from lib.lsp.solver_lsp import test_lsp

# return a list of loss computed on a specified grid over the solution path
def get_losses_lsp(lam_min, lam_max, num_grid, model, data_loader, loss_fn, device='cpu'):
    lambdas = np.linspace(lam_max, lam_min, num_grid)
    losses = []
    for lam in lambdas:
        losses.append(test_lsp(data_loader, model, loss_fn, lam, device))
            
    return losses
    
# return the absolute errors compared to the true loss accross the solution path  
def get_errs_lsp(lam_min, lam_max, true_loss_list, model, data_loader, loss_fn, device='cpu'):
    losses = get_losses_lsp(lam_min, lam_max, len(true_loss_list), model, data_loader, loss_fn, device)
    return losses - true_loss_list
    
# return the supremum absolute error compared to the true loss accross the solution path  
def get_sup_error_lsp(lam_min, lam_max, true_loss_list, model, data_loader, loss_fn, device='cpu'):
    errs = get_errs_lsp(lam_min, lam_max, true_loss_list, model, data_loader, loss_fn, device)
    return max(errs)

# return a 2-d numpy array of loss computed on a specified grid over the solution path
def get_losses_lsp_2d(lam_min_2d, lam_max_2d, num_grid_2d, model, data_loader, loss_fn, device='cpu'):
    lambdas_0 = np.linspace(lam_max_2d[0], lam_min_2d[0], num_grid_2d[0])
    losses_2d = []
    for fix_lam in lambdas_0:
        lambdas_1 = np.linspace(lam_max_2d[1], lam_min_2d[1], num_grid_2d[1])
        losses = []
        for running_lam in lambdas_1:
            losses.append(test_lsp(data_loader, model, loss_fn, [fix_lam, running_lam], device))
        losses_2d.append(losses)
            
    return np.array(losses_2d)

# input true_loss_list must be numpy array
def get_errs_lsp_2d(lam_min, lam_max, true_loss_list, model, data_loader, loss_fn, device='cpu'):
    losses = get_losses_lsp_2d(lam_min, lam_max, [len(true_loss_list), len(true_loss_list[0])], model, data_loader, loss_fn, device)
    return losses - true_loss_list
    
# return the supremum absolute error compared to the true loss accross the 2-d solution path  
def get_sup_error_lsp_2d(lam_min, lam_max, true_loss_list, model, data_loader, loss_fn, device='cpu'):
    errs_2d = get_errs_lsp_2d(lam_min, lam_max, true_loss_list, model, data_loader, loss_fn, device)
    return np.max(np.array(errs_2d))