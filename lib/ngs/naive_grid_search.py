import numpy as np
import torch
from lib.ngs.log_reg_module import Logistic_Regression
from lib.ngs.reg_solver import train, test

"""# Naive Grid Search"""

# running gradient descent with fixed learning rate on a single grid point, i.e. for one specified lambda
def GD_on_a_grid(lam, lam_max, epochs, loss_fn, model, avg_model, optimizer, trainDataLoader,
                 step_size=None, const=None, weighted_avg=False, testDataLoader=None, 
                 oracle=False, true_loss_list=None, fine_delta_lam=None, stopping_criterion=None, 
                 check_frequency=5, device="cpu"):
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
                   
    if weighted_avg:
        avg_model.reg_param = lam
        avg_weight = model.linear.weight.clone().detach().squeeze() # weighted averaging sum initialized
        avg_intercept = model.linear.bias.clone().detach().squeeze()

    early_stop = False
    itr = 0
    passes = 0
    error = 0
                   
    for t in range(epochs):

        itr, avg_weight, avg_intercept = train(itr, avg_weight, avg_intercept, 
                                               trainDataLoader, model, loss_fn, optimizer, device)
      
        if (true_loss_list is not None) and (oracle):
            if (t+1) % check_frequency == 0:
                # do an accuracy check
                if weighted_avg:
                    with torch.no_grad():
                        avg_model.linear.weight.copy_(avg_weight)
                        avg_model.linear.bias.copy_(avg_intercept)
                    approx_loss = test(testDataLoader, avg_model, loss_fn, lam, device)
                else:
                    approx_loss = test(testDataLoader, model, loss_fn, lam, device)
                    
                error = approx_loss - true_loss
                # check if we are within stopping criterion
                if error <= stopping_criterion:
                    passes += (t+1)
                    early_stop = True
                    break  # Early stop
            
    if not early_stop:
        passes += epochs
        
    return passes, error, avg_weight, avg_intercept

"""Naive Grid Search starts from $\lambda = 1$ and decreases $\lambda$ by $\Delta\lambda = \frac{\lambda_\text{max} - \lambda_\text{min}}{\text{# of grid}}$. The model trained on each grid point $(\lambda - \Delta\lambda)$ initializes weight with the linear weight of the model trained on the previous grid point $\lambda$."""

# do the whole naive grid search over a list of uniformly spaced lambda's
# from lam_min to lam_max
# returns a list of trained models
def naive_grid_search(lam_min, lam_max, num_grid, epochs, loss_fn, trainDataLoader,
                      data_input_dim, lr=1e-3, step_size=None, const=None, weighted_avg=False, 
                      testDataLoader=None, oracle=False, true_loss_list=None, stopping_criterion=None, 
                      check_frequency=5, device="cpu"):
    fine_delta_lam = None
    if true_loss_list is not None:
        fine_delta_lam = (lam_max - lam_min)/(len(true_loss_list) - 1)
        
    reg_params = []
    weights = []
    intercepts = []
    avg_weights = []
    avg_intercepts = []
    total_passes = 0
    grid_pass_error = 0
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
        passes, grid_error, avg_weight, avg_intercept = GD_on_a_grid(lam, lam_max, epochs, loss_fn, 
                                                         model, avg_model, optimizer,
                                                         trainDataLoader=trainDataLoader,
                                                         step_size=step_size, const=const, 
                                                         weighted_avg=weighted_avg,
                                                         testDataLoader=testDataLoader,
                                                         oracle=oracle,
                                                         true_loss_list=true_loss_list,
                                                         fine_delta_lam=fine_delta_lam,
                                                         stopping_criterion=stopping_criterion,
                                                         check_frequency=check_frequency,
                                                         device=device)
        if weighted_avg:
            avg_weights.append(avg_weight)
            avg_intercepts.append(avg_intercept)
        else:
            weights.append(model.linear.weight.clone().data.cpu().squeeze().numpy())
            intercepts.append(model.linear.bias.clone().data.cpu().squeeze().numpy())                  

        reg_params.append(model.reg_param)
        total_passes += passes
        grid_pass_error = max([grid_pass_error, grid_error])

    if weighted_avg:
        return total_passes, reg_params, avg_intercepts, avg_weights, grid_pass_error
    else:
        return total_passes, reg_params, intercepts, weights, grid_pass_error
