import torch
from lib.lsp.basis_tf_module import Basis_TF_SGD
from lib.lsp.reg_solver_lsp import train_lsp
from lib.lsp.utils_lsp import get_sup_error_lsp

# step_size function take in 2 parameters: current iteration number, and a self-defined constant 
# step_size function returns the learning rate for current iteration
# make sure to input 'distribution' or else the 
def learn_solution_path(input_dim, basis_dim, phi_lam, epochs, trainDataLoader, testDataLoader, loss_fn, lam_min, lam_max, 
                        true_losses, lr=1e-3, step_size=None, const=None, SGD=False, init_weight=None, 
                        intercept=True, weighted_avg=False, record_frequency=10, distribution='uniform', device='cpu', trace_frequency=-1):
    # build the model
    model = Basis_TF_SGD(input_dim, basis_dim, phi_lam, init_weight=init_weight, intercept=intercept).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
                          
    # initialize weighted averaging sum
    if weighted_avg:
        avg_model = Basis_TF_SGD(input_dim, basis_dim, phi_lam, init_weight=init_weight, intercept=intercept).to(device)
        avg_weight = avg_model.linear.weight.clone().detach()
        sup_err_history_avg = []
                          
    sup_err_history = []
    num_itr_history = []

    for t in range(epochs):
        if step_size is not None:
            # shrink learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = step_size(t, const)
                
        train_lsp(trainDataLoader, model, loss_fn, optimizer, distribution, device)

        if weighted_avg and t>50: 
            rho = 2 / (t+3)
            avg_weight = (1-rho) * avg_weight + rho * model.linear.weight.clone().detach()
        else:
            avg_weight = model.linear.weight.clone().detach()
            
        if (t+1) % record_frequency == 0:
            num_itr_history.append(t+1)
            if weighted_avg:
                with torch.no_grad():
                    avg_model.linear.weight.copy_(avg_weight)
                sup_err = get_sup_error_lsp(lam_min, lam_max, true_losses,
                                            avg_model, testDataLoader, loss_fn, device)
            else:
                sup_err = get_sup_error_lsp(lam_min, lam_max, true_losses,
                                            model, testDataLoader, loss_fn, device)
            sup_err_history.append(sup_err)
            if (trace_frequency > 0) & ((t+1) % trace_frequency == 0):
                print(f"--------approximate solution path for # itr = {t+1} complete--------")
                print(f"# itr: {t+1}\t sup error: {sup_err}")
    if weighted_avg:
        return num_itr_history, sup_err_history, avg_weight
    else:
        return num_itr_history, sup_err_history, model.linear.weight.clone().detach()
    
