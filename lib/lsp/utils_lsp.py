import numpy as np
from lib.lsp.reg_solver_lsp import test_lsp

# return a list of loss computed on a specified grid over the solution path
def get_losses_SGD(model, lam_min, lam_max, num_grid, data_loader, loss_fn, device='cpu'):
    lambdas = np.linspace(lam_max, lam_min, num_grid)
    losses = []
    for lam in lambdas:
        losses.append(test_lsp(data_loader, model, loss_fn, lam, device))
            
    return losses
    
# return the absolute errors compared to the true loss accross the solution path  
def get_errs_SGD(model, lam_min, lam_max, true_loss_list, data_loader, loss_fn, device='cpu'):
    losses = get_losses_SGD(model, lam_min, lam_max, len(true_loss_list), data_loader, loss_fn, device)
    return losses - true_loss_list
    
# return the supremum absolute error compared to the true loss accross the solution path  
def get_sup_error_SGD(lam_min, lam_max, true_loss_list, model, data_loader, loss_fn, device='cpu'):
    errs = get_errs_SGD(model, lam_min, lam_max, true_loss_list, data_loader, loss_fn, device)
    return max(errs)
