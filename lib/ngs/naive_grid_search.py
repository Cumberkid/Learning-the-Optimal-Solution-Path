import numpy as np
import torch
from lib.ngs.log_reg_module import Logistic_Regression
from lib.ngs.reg_solver import train, test

"""# Naive Grid Search"""

# running gradient descent with fixed learning rate on a single grid point, i.e. for one specified lambda
def GD_on_a_grid(lam, lam_max, epochs, loss_fn, model, avg_model, optimizer, trainDataLoader,
                 step_size=None, const=1, SGD=False,
                 testDataLoader=None, true_loss_list=None, fine_delta_lam=None, stopping_criterion=None, 
                 record_frequency=5, device="cpu"):
    # performs early-stop if the true solution path is known                
    if true_loss_list is not None:
        # true loss
        i = round((lam_max - lam) / fine_delta_lam)
        if i >= len(true_loss_list):
            i = len(true_loss_list) - 1
        true_loss = true_loss_list[i]
        lam = lam_max - i * fine_delta_lam
        # print(f"nearest i = {i}\t lam = {lam}")

    model.reg_param = lam
                   
    if SGD:
        avg_model.reg_param = lam
        avg_weight = model.linear.weight.clone().detach()[0] # weighted averaging sum initialized
        avg_intercept = model.linear.bias.clone().detach()[0]

    early_stop = False
    itr = 0
                   
    for t in range(epochs):
        # if SGD and (t+2 > t_0):
        if SGD:
            # shrink learning rate:
            for param_group in optimizer.param_groups:
                param_group['lr'] = step_size(t, const)

        train(trainDataLoader, model, loss_fn, optimizer, device)
      
        if SGD:
            rho = 2 / (t+3) #compute weighted averaging sum
            avg_weight = (1-rho) * avg_weight + rho * model.linear.weight.clone().detach()[0]
            avg_intercept = (1-rho) * avg_intercept + rho * model.linear.bias.clone().detach()[0]
      
        if true_loss_list is not None:
            if (t+1) % record_frequency == 0:
                # do an accuracy check
                if SGD:
                    with torch.no_grad():
                        avg_model.linear.weight.copy_(avg_weight)
                        avg_model.linear.bias.copy_(avg_intercept)
                    approx_loss = test(testDataLoader, avg_model, loss_fn, lam, device)
                else:
                    approx_loss = test(testDataLoader, model, loss_fn, lam, device)
                    
                error = approx_loss - true_loss
                # stopping criterion
                if error <= stopping_criterion:
                    itr += (t+1)
                    early_stop = True
                    break  # Early stop
            
    if not early_stop:
        itr += epochs
        
    return itr

"""Naive Grid Search starts from $\lambda = 1$ and decreases $\lambda$ by $\Delta\lambda = \frac{\lambda_\text{max} - \lambda_\text{min}}{\text{# of grid}}$. The model trained on each grid point $(\lambda - \Delta\lambda)$ initializes weight with the linear weight of the model trained on the previous grid point $\lambda$."""

# do the whole naive grid search over a list of uniformly spaced lambda's
# from lam_min to lam_max
# returns a list of trained models
def naive_grid_search(lam_min, lam_max, num_grid, epochs, loss_fn, trainDataLoader,
                      data_input_dim, lr=1e-3, step_size=None, const=1, SGD=False, 
                      testDataLoader=None, true_loss_list=None, stopping_criterion=None, 
                      record_frequency=5, device="cpu"):
    fine_delta_lam = None
    if true_loss_list is not None:
        fine_delta_lam = (lam_max - lam_min)/(len(true_loss_list) - 1)
        
    reg_params = []
    weights = []
    intercepts = []
    avg_weights = []
    avg_intercepts = []
    total_itr = 0
    # create a list of lambda's
    lambdas = np.linspace(lam_max, lam_min, num_grid)
    
    # first weight is initialized at 0
    weight = torch.zeros(data_input_dim)
    intercept = 0
    model = Logistic_Regression(data_input_dim, 1, lam_max, weight, intercept).to(device)
    avg_model = Logistic_Regression(data_input_dim, 1, lam_max, weight, intercept).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    
    for lam in lambdas:
        # print(f"Running model on lambda = {lam}")
        itr = GD_on_a_grid(lam, lam_max, epochs, loss_fn, model, avg_model, optimizer,
                           trainDataLoader=trainDataLoader,
                           step_size=step_size,
                           const=const, SGD=SGD,
                           testDataLoader=testDataLoader,
                           true_loss_list=true_loss_list,
                           fine_delta_lam=fine_delta_lam,
                           stopping_criterion=stopping_criterion,
                           record_frequency=record_frequency,
                           device=device)
        if SGD:
            avg_weights.append(avg_model.linear.weight.clone().data.cpu().numpy()[0])
            avg_intercepts.append(avg_model.linear.bias.clone().data.cpu().numpy()[0])
        else:
            weights.append(model.linear.weight.clone().data.cpu().numpy()[0])
            intercepts.append(model.linear.bias.clone().data.cpu().numpy()[0])                  
        # print(model.linear.weight)
        reg_params.append(model.reg_param)
        total_itr += itr
        # print(total_itr)

    if SGD:
        return total_itr, reg_params, avg_intercepts, avg_weights
    else:
        return total_itr, reg_params, intercepts, weights
