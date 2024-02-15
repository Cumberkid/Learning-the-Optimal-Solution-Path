import numpy as np
from lib.lsp.log_reg_solver_sgd import test_SGD
from lib.lsp.fair_reg_solver_sgd import fair_test_SGD

# return a list of loss computed on a specified grid over the solution path
def get_losses_SGD(model, lam_min, lam_max, num_grid, data_loader, loss_fn, obj=None):
    if obj is None:
        print("Please enter the objective: 'logit' or 'fairness'")
        return
    lambdas = np.linspace(lam_max, lam_min, num_grid)
    losses = []
    for lam in lambdas:
        if obj == "logit":
            losses.append(test_SGD(data_loader, model, loss_fn, lam))
        elif obj == "fairness":
            losses.append(fair_test_SGD(data_loader, model, loss_fn, lam))
            
    return losses
    
# return the absolute errors compared to the true loss accross the solution path  
def get_errs_SGD(model, lam_min, lam_max, true_loss_list, data_loader, loss_fn, obj=None):
    if obj is None:
        print("Please enter the objective: 'logit' or 'fairness'")
        return
    losses = get_losses_SGD(model, lam_min, lam_max, len(true_loss_list), data_loader, loss_fn, obj=obj)
    return losses - true_loss_list
    
# return the supremum absolute error compared to the true loss accross the solution path  
def get_sup_error_SGD(lam_min, lam_max, true_loss_list, model, data_loader, loss_fn, obj=None):
    if obj is None:
        print("Please enter the objective: 'logit' or 'fairness'")
        return
    errs = get_errs_SGD(model, lam_min, lam_max, true_loss_list, data_loader, loss_fn, obj=obj)
    return max(errs)
