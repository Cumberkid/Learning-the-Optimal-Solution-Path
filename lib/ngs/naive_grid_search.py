import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from lib.ngs.log_reg_module import Logistic_Regression
from lib.ngs.log_reg_solver import train, test
from lib.ngs.fair_reg_solver import fair_train, fair_test

"""# Naive Grid Search"""

# running gradient descent with fixed learning rate on a single grid point, i.e. for one specified lambda
def GD_on_a_grid(lam, lam_max, epochs, loss_fn, model, optimizer, trainDataLoader, data_input_dim,
                 obj=None, alpha=1, init_lr=0.1, diminish=False, gamma=0.1, dim_step=30, SGD=False, 
                 testDataLoader=None, true_loss_list=None, fine_delta_lam=None, stopping_criterion=None, device="cpu"):
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
    if diminish:
        # Define the learning rate scheduler
        scheduler = StepLR(optimizer, step_size=dim_step, gamma=gamma)  # Decrease LR by a factor of gamma every dim_step epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = init_lr  
        
    early_stop = False
    itr = 0
    for t in range(epochs):
        if SGD:
            # shrink learning rate:
            lr = min([init_lr, alpha/(t+1)])
            optimizer.zero_grad()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
        if obj == "logit":
            train(trainDataLoader, model, loss_fn, optimizer, device)
        elif obj == "fairness":
            fair_train(trainDataLoader, model, loss_fn, optimizer, device)
            
        if true_loss_list is not None:
            if (t+1) % 10 == 0:
                # do an accuracy check
                if obj == "logit":
                    approx_loss = test(testDataLoader, model, loss_fn, lam, device)
                elif obj == "fairness":
                    approx_loss = fair_test(testDataLoader, model, loss_fn, lam, device)
                    
                error = approx_loss - true_loss
                # print(lr, error, true_loss)
                # stopping criterion
                if error <= stopping_criterion:
                    itr += (t+1)
                    early_stop = True
                    break  # Early stop
                
        if diminish:
            # Update the learning rate
            scheduler.step()
            
    if not early_stop:
        itr += epochs
        
    return itr

"""Naive Grid Search starts from $\lambda = 1$ and decreases $\lambda$ by $\Delta\lambda = \frac{\lambda_\text{max} - \lambda_\text{min}}{\text{# of grid}}$. The model trained on each grid point $(\lambda - \Delta\lambda)$ initializes weight with the linear weight of the model trained on the previous grid point $\lambda$."""

# do the whole naive grid search over a list of uniformly spaced lambda's
# from lam_min to lam_max
# returns a list of trained models
def naive_grid_search(lam_min, lam_max, num_grid, epochs, loss_fn, trainDataLoader,
                      data_input_dim, obj=None, lr=1e-3, alpha=1, init_lr=1, 
                      diminish=False, gamma=0.1, dim_step=30, SGD=False,
                      testDataLoader=None, true_loss_list=None, stopping_criterion=None, device="cpu"):
    if obj is None:
        print("Please enter the objective: 'logit' or 'fairness'")
        return
    
    fine_delta_lam = None
    if true_loss_list is not None:
        fine_delta_lam = (lam_max - lam_min)/(len(true_loss_list) - 1)
        
    reg_params = []
    weights = []
    intercepts = []
    total_itr = 0
    # create a list of lambda's
    lambdas = np.linspace(lam_max, lam_min, num_grid)
    
    # first weight is initialized at 0
    weight = torch.zeros(data_input_dim)
    intercept = 0
    model = Logistic_Regression(data_input_dim, 1, lam_max, weight, intercept).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    
    for lam in lambdas:
        # print(f"Running model on lambda = {lam}")
        itr = GD_on_a_grid(lam, lam_max, epochs, loss_fn, model, optimizer,
                                  trainDataLoader=trainDataLoader,
                                  data_input_dim=data_input_dim,
                                  obj=obj, alpha=alpha, 
                                  init_lr=init_lr, diminish=diminish, 
                                  gamma=gamma, dim_step=dim_step, SGD=SGD, 
                                  testDataLoader=testDataLoader,
                                  true_loss_list=true_loss_list,
                                  fine_delta_lam=fine_delta_lam,
                                  stopping_criterion=stopping_criterion,
                                  device=device)
        weights.append(model.linear.weight.clone().data.cpu().numpy()[0])
        intercepts.append(model.linear.bias.clone().data.cpu().numpy()[0])
        # print(model.linear.weight)
        reg_params.append(model.reg_param)
        total_itr += itr
        # print(total_itr)
        
    return total_itr, reg_params, intercepts, weights
