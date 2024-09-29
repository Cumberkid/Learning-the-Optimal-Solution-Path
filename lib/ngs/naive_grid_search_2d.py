import numpy as np
import torch
from lib.ngs.solver import train, test
from lib.ngs.portfolio_alloc_module import Portfolio_Allocation

"""# Naive Grid Search"""

# running gradient descent with fixed learning rate on a single grid point, i.e. for one specified lambda
def GD_on_a_grid(fix_lam, running_lam, lam_max, epochs, loss_fn, model, test_model, optimizer, trainDataLoader,
                 step_size=None, const=None, weighted_avg=False, testDataLoader=None, 
                 oracle=False, true_loss_list=None, fine_delta_lam=None, stopping_criterion=None, 
                 check_frequency=5, device="cpu"):

    # performs early-stop if the true solution path is known                
    if true_loss_list is not None:
        # true loss
        i = round((lam_max - running_lam) / fine_delta_lam)
        if i >= len(true_loss_list):
            i = len(true_loss_list) - 1
        true_loss = true_loss_list[i]
        running_lam = lam_max - i * fine_delta_lam
        # print(f"nearest i = {i}\t lam = {lam}")

    model.hyper_param = [fix_lam, running_lam]
    weight = model.linear.weight.clone().detach().squeeze() # weighted averaging sum initialized
    intercept = 0
    if model.bias is not None:
        intercept = model.linear.bias.clone().detach().squeeze()

    early_stop = False
    itr = 0
    passes = 0
    error = 0
                   
    for t in range(epochs):

        itr, weight, intercept = train(itr, weight, intercept, trainDataLoader, model, loss_fn, 
                                       optimizer, weighted_avg, step_size, const, device)
        # print(model.hyper_param, weight)
        if true_loss_list is not None:
            if (t+1) % check_frequency == 0:
                # do an accuracy check
                test_model.hyper_param = model.hyper_param
                with torch.no_grad():
                    test_model.linear.weight.copy_(weight)
                    if model.bias is not None:
                        test_model.linear.bias.copy_(intercept)
                approx_loss = test(testDataLoader, test_model, loss_fn, test_model.hyper_param, device)
                    
                error = approx_loss - true_loss
                
                # check if we are within stopping criterion
                if oracle:
                  if error <= stopping_criterion:
                      passes += (t+1)
                      early_stop = True
                      break  # Early stop
            
    if not early_stop:
        passes += epochs
        
    return passes, error, weight, intercept

"""Naive Grid Search starts from $\lambda = 1$ and decreases $\lambda$ by $\Delta\lambda = \frac{\lambda_\text{max} - \lambda_\text{min}}{\text{# of grid}}$. The model trained on each grid point $(\lambda - \Delta\lambda)$ initializes weight with the linear weight of the model trained on the previous grid point $\lambda$."""

# do the whole naive grid search over a list of uniformly spaced lambda's
# from lam_max to lam_min
# returns a list of trained models
def naive_grid_search(fix_lam, lam_min, lam_max, num_grid, epochs, loss_fn, trainDataLoader,
                      data_input_dim, lr=1e-3, step_size=None, const=None, weighted_avg=False, 
                      testDataLoader=None, oracle=False, true_loss_list=None, stopping_criterion=None, 
                      check_frequency=5, device="cpu"):
    fine_delta_lam = None
    if true_loss_list is not None:
        fine_delta_lam = (lam_max - lam_min)/(len(true_loss_list) - 1)
        
    hyper_params = []
    weights = []
    intercepts = []
    total_passes = 0
    grid_pass_error = 0
    # create a list of lambda's
    lambdas = np.linspace(lam_max, lam_min, num_grid)
    
    # first weight is initialized at 0
    weight = torch.zeros(data_input_dim)
    intercept = 0
    model = Portfolio_Allocation(data_input_dim, 1, [fix_lam, lam_max], weight).to(device)
    test_model = Portfolio_Allocation(data_input_dim, 1, [fix_lam, lam_max], weight).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    
    for running_lam in lambdas:
        # print(f"Running model on lambda = {running_lam}")
        passes, grid_error, weight, intercept = GD_on_a_grid(fix_lam, running_lam, lam_max, epochs, 
                                                            loss_fn, model, test_model, optimizer,
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
        # print(running_lam, weight)
        weights.append(weight)
        intercepts.append(intercept)               

        hyper_params.append(model.hyper_param)
        total_passes += passes
        grid_pass_error = max([grid_pass_error, grid_error])

    return total_passes, hyper_params, intercepts, weights, grid_pass_error



def naive_grid_search_2d(lam_min_2d, lam_max_2d, num_grid_2d, epochs, loss_fn, trainDataLoader,
                      data_input_dim, lr=1e-3, step_size=None, const=None, weighted_avg=False, 
                      testDataLoader=None, oracle=False, true_loss_list=None, stopping_criterion=None, 
                      check_frequency=5, device="cpu"):
    
    total_passes_2d = 0
    hyper_params_2d = []
    weights_2d = []
    intercepts_2d = []
    grid_pass_error_2d = 0

    # create a list on the first hyper parameter
    lambdas = np.linspace(lam_max_2d[0], lam_min_2d[0], num_grid_2d[0])
    i = 0
    running_list = None
    for fix_lam in lambdas:
        if true_loss_list is not None:
            running_list = true_loss_list[i]
            i += 1

        total_passes, hyper_params, intercepts, weights, grid_pass_error = naive_grid_search(fix_lam, lam_min_2d[1], lam_max_2d[1], num_grid_2d[1], 
                                                                                             epochs, loss_fn, trainDataLoader,
                                                                                             data_input_dim, lr=lr, step_size=step_size, 
                                                                                             const=const, weighted_avg=weighted_avg, 
                                                                                             testDataLoader=testDataLoader, oracle=oracle, 
                                                                                             true_loss_list=running_list, 
                                                                                             stopping_criterion=stopping_criterion, 
                                                                                             check_frequency=check_frequency, device=device)
        
        total_passes_2d += total_passes
        hyper_params_2d.append(hyper_params)
        weights_2d.append(weights)
        intercepts_2d.append(intercepts)
        grid_pass_error_2d = max([grid_pass_error, grid_pass_error_2d])

    return total_passes_2d, hyper_params_2d, intercepts_2d, weights_2d, grid_pass_error_2d