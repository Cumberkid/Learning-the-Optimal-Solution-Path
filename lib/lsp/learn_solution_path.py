import torch
import math
from collections import deque
from lib.lsp.basis_tf_module import Basis_TF_SGD
from lib.lsp.solver_lsp import train_lsp, train_lsp_bfgs, test_lsp
from lib.lsp.utils_lsp import get_sup_error_lsp
from lib.utils import gauss_legendre_integral, gauss_legendre_integral_2d, monte_carlo
import itertools
from scipy.special import beta as beta_func


def distance(init_weight, prev_weight, curr_weight, n, k, thresh=0.6, q=1.5, k_0=5):
    if n == int(q**k):
        # print(n)
        if k >= k_0:
            s = (torch.log((init_weight - curr_weight).norm(p=2)**2) - torch.log((init_weight - prev_weight).norm(p=2)**2)) / (math.log(n) - math.log(n/q))
            diag = (s < thresh)
        else:
            diag = False
        while n == int(q**k):
            k += 1
        prev_weight = curr_weight
    else:
        diag = False

    return diag, k, prev_weight

# Default for diminishing stepsize is to use the 'distance diagnostic' described in this paper:
# Returns the list of epochs with corresponding solution path errors, final weight of the trained model for the modified stochastic
# optimization problem, final learning rate if use distance diagnostic, and total number of iterations
# lam_min and lam_max must be list objects
def learn_solution_path(input_dim, basis_dim, phi_lam, max_epochs, trainDataLoader, testDataLoader, loss_fn, lam_min, lam_max,
                        true_losses=None, init_lr=1e-3, diminish=False, gamma=0.97, q=1.3, k_0=5, thresh_lr=0.6,
                        step_size=None, const=None, init_weight=None, intercept=True, weighted_avg=False, itr=0, thresh_basis=None,
                        record_frequency=10, record_fctn=get_sup_error_lsp, opt_method='SGD', distribution='uniform', 
                        alpha_beta=[-.5, -.5], device='cpu', trace_frequency=-1):
    # build the model
    model = Basis_TF_SGD(input_dim, basis_dim, phi_lam, init_weight=init_weight, intercept=intercept).to(device)
    lr=init_lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if opt_method == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_method == 'lbfgs':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
    optimizer.zero_grad()

    # memorize the previous stochastic objective value to decide if need to add more basis function
    stochastic_obj_list = deque(maxlen=2)

    # # memorize the last 10 gradient info to decide if need to add more basis function
    # norm_grad_list = deque(maxlen=20)

    # initialize weighted averaging sum
    if weighted_avg:
        avg_model = Basis_TF_SGD(input_dim, basis_dim, phi_lam, init_weight=init_weight, intercept=intercept).to(device)


    sup_err_history = []
    num_pass_history = []

    if diminish:
        # initilize for distance diagnostic
        init_weight = model.linear.weight.clone().detach() # record inital weight theta_0
        prev_weight = model.linear.weight.clone().detach() # memorize theta_n/q for every n=q^k until max_epochs/q
        k = 0

    if thresh_basis is None:
        thresh_basis = lambda x: 1e-4

    itr = itr
    weight = model.linear.weight.clone().detach()
    
    for t in range(max_epochs):
        # run one pass of dataset
        if (opt_method == 'SGD') or (opt_method == 'Adam'):
            new_grad, weight, itr = train_lsp(itr, weight, trainDataLoader, model, loss_fn, optimizer, 
                                            lam_min=lam_min, lam_max=lam_max, 
                                            weighted_avg=weighted_avg, step_size=step_size, 
                                            const=const, distribution=distribution, alpha_beta=alpha_beta, device=device)
        elif opt_method == 'lbfgs':
            new_grad, weight, itr = train_lsp_bfgs(itr, weight, trainDataLoader, model, loss_fn, optimizer, 
                                            lam_min=lam_min, lam_max=lam_max, 
                                            step_size=step_size, 
                                            const=const, distribution=distribution, device=device)

        if diminish: # diminish according to distance diagnostic
            # run distance diagnostic
            curr_weight = model.linear.weight.clone().detach()
            distance_diag, k, prev_weight = distance(init_weight, prev_weight, curr_weight, t+1, k, thresh_lr, q, k_0)
            if distance_diag:
                lr = gamma * lr
                # print(f"diminish at iteration #{t+1}, new lr = {lr}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        # record iteration result
        if (record_frequency > 0) & ((t+1) % record_frequency == 0):
            if (opt_method == 'SGD') or (opt_method == 'Adam'):
                num_pass_history.append(t+1)
            elif opt_method == 'lbfgs':
                num_pass_history.append(itr)
            if weighted_avg:
                with torch.no_grad():
                    avg_model.linear.weight.copy_(weight)
                sup_err = record_fctn(lam_min, lam_max, true_losses,
                                            avg_model, testDataLoader, loss_fn, device)
            else:
                sup_err = record_fctn(lam_min, lam_max, true_losses,
                                            model, testDataLoader, loss_fn, device)
            sup_err_history.append(sup_err)
            if (trace_frequency > 0) & ((t+1) % trace_frequency == 0):
                if (opt_method == 'SGD') or (opt_method == 'Adam'):
                    print(f"--------approximate solution path for # itr = {t+1} complete--------")
                    
                elif opt_method == 'lbfgs':
                    print(f"--------approximate solution path for # itr = {itr} complete--------")
                
                print(f"# epoch: {t+1}\t sup error: {sup_err}")

        # when the change in the stochastic objective is small enough,
        # stop to add more basis functions
        def func(hyper_params):
            return test_lsp(testDataLoader, model, loss_fn, hyper_params, device)
        stochastic_obj = 0.0
        num_quadpts = 10
        num_pts = 1000
        if len(lam_min) == 1:
            stochastic_obj = gauss_legendre_integral(func, lam_min[0], lam_max[0], num_quadpts)
        elif len(lam_min) == 2:
            stochastic_obj = gauss_legendre_integral_2d(func, lam_min, lam_max, num_quadpts)
        elif len(lam_min) > 2:
            stochastic_obj = monte_carlo(func, lam_min, lam_max, num_pts, distribution, 88)
            print(f"--------approximate solution path for # function evaluation = {itr} complete--------")
            print(f"# gradient call: {t+1}\t average loss: {stochastic_obj}")
        stochastic_obj_list.append(stochastic_obj)
        if (len(stochastic_obj_list) >= 2):
            if abs(stochastic_obj_list[1] - stochastic_obj_list[0]) < thresh_basis(basis_dim):
                break

        # # when the change in second moment of gradient is small enough,
        # # stop to add more basis functions
        # norm_grad_list.append(new_grad.norm(p=2)**2)
        # if (len(norm_grad_list) >= 20) and abs(sum(list(norm_grad_list)[:10]) - sum(list(norm_grad_list)[-10:])) < thresh_basis(basis_dim):
        #     break
        # if opt_method == 'lbfgs':
        #     if (len(norm_grad_list) >= 2) and abs(norm_grad_list[len(norm_grad_list) - 1] - norm_grad_list[len(norm_grad_list) - 2]) < thresh_basis(basis_dim):
        #         break

    return num_pass_history, sup_err_history, weight, lr, itr
    

def adaptive_lsp(input_dim, start_basis_dim, end_basis_dim, phi_lam, max_epochs,
                 trainDataLoader, testDataLoader, loss_fn, lam_min, lam_max, true_losses=None,
                 init_lr=1e-3, diminish=True, gamma=0.97, q=1.3, k_0=5, thresh_lr=0.6,
                 step_size=None, const=None, init_weight=None, intercept=True, weighted_avg=False,
                 avg_upon_prev=False, thresh_basis=None, record_frequency=10, record_fctn=get_sup_error_lsp, 
                 opt_method='SGD', distribution='uniform', alpha_beta=[-.5, -.5], 
                 device='cpu', basis_list=None, lr_list=None, trace_frequency=-1):
    if step_size is not None:
        diminish = False

    num_pass_history = []
    sup_err_history = []
    breaks = []
    weight = init_weight
    lr = init_lr
    break_itr = 0
    const = const
    itr = 0

    if opt_method=='lbfgs':
        break_grads = []
        
    if basis_list is not None and lr_list is not None:
        iter1, iter2 = itertools.tee(basis_list)
        next(iter2)  # Advance the second iterator by 1 to access the next element

        for current_dim, next_dim, lr in zip(iter1, iter2, lr_list):
            print(f"********** now running lsp with #basis dimension = {current_dim} ***********")
            print(f'lr = {lr}')
            num_pass_current, sup_err_current, weight, lr, itr = learn_solution_path(input_dim, current_dim, phi_lam, 
                                max_epochs, trainDataLoader, testDataLoader, loss_fn, lam_min, lam_max, true_losses, 
                                init_lr=lr, diminish=diminish, gamma=gamma, q=q, k_0=k_0, thresh_lr=thresh_lr,
                                step_size=step_size, const=const, init_weight=weight, intercept=intercept, 
                                weighted_avg=weighted_avg, itr=itr, thresh_basis=thresh_basis, 
                                record_frequency=record_frequency, record_fctn=record_fctn, opt_method=opt_method, 
                                distribution=distribution, alpha_beta=alpha_beta, device=device, 
                                trace_frequency=trace_frequency)

            num_pass_history += [(x + break_itr) for x in num_pass_current]
            sup_err_history += sup_err_current
            break_itr = num_pass_history[len(num_pass_history)-1]
            breaks.append(break_itr)
            if opt_method=='lbfgs':
                break_grads.append(len(num_pass_history))
            if const is not None:
                const = const/(2**0.5)
            if not avg_upon_prev:
                itr = 0
            # increase basis dimension by 1
            weight = torch.cat((weight, torch.zeros(weight.size(dim=0), next_dim - current_dim)), 1)

    else:
        for basis_dim in range(start_basis_dim, end_basis_dim):
            print(f"********** now running lsp with #basis dimension = {basis_dim} ***********")
            print(f'lr = {lr}')
            num_pass_current, sup_err_current, weight, lr, itr = learn_solution_path(input_dim, basis_dim, phi_lam, 
                                max_epochs, trainDataLoader, testDataLoader, loss_fn, lam_min, lam_max, true_losses, 
                                init_lr=lr, diminish=diminish, gamma=gamma, q=q, k_0=k_0, thresh_lr=thresh_lr,
                                step_size=step_size, const=const, init_weight=weight, intercept=intercept, 
                                weighted_avg=weighted_avg, itr=itr, thresh_basis=thresh_basis, 
                                record_frequency=record_frequency, record_fctn=record_fctn, opt_method=opt_method, 
                                distribution=distribution, alpha_beta=alpha_beta, device=device, 
                                trace_frequency=trace_frequency)

            num_pass_history += [(x + break_itr) for x in num_pass_current]
            sup_err_history += sup_err_current
            break_itr = num_pass_history[len(num_pass_history)-1]
            breaks.append(break_itr)
            if const is not None:
                const = const/(2**0.5)
            if not avg_upon_prev:
                itr = 0
            # increase basis dimension by 1
            weight = torch.cat((weight, torch.zeros(weight.size(dim=0), 1)), 1)

    if opt_method=='lbfgs':
        return num_pass_history, sup_err_history, breaks, break_grads

    return num_pass_history, sup_err_history, breaks







def learn_solution_path_high_dim(input_dim, basis_dim, phi_lam, max_epochs, trainDataLoader, testDataLoader, loss_fn_train, loss_fn_val, lam_min, lam_max,
                        true_losses=None, init_lr=1e-3, diminish=False, gamma=0.97, q=1.3, k_0=5, thresh_lr=0.6,
                        step_size=None, const=None, init_weight=None, intercept=True, weighted_avg=False, itr=0, thresh_basis=None,
                        record_frequency=10, record_fctn=get_sup_error_lsp, opt_method='SGD', distribution='uniform', 
                        alpha_beta=[-.5, -.5], device='cpu', trace_frequency=-1):
    # build the model
    model = Basis_TF_SGD(input_dim, basis_dim, phi_lam, init_weight=init_weight, intercept=intercept).to(device)
    lr=init_lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if opt_method == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_method == 'lbfgs':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
    optimizer.zero_grad()

    # memorize the previous stochastic objective value to decide if need to add more basis function
    stochastic_obj_list = deque(maxlen=2)

    # # memorize the last 10 gradient info to decide if need to add more basis function
    # norm_grad_list = deque(maxlen=20)

    # initialize weighted averaging sum
    if weighted_avg:
        avg_model = Basis_TF_SGD(input_dim, basis_dim, phi_lam, init_weight=init_weight, intercept=intercept).to(device)


    sup_err_history = []
    num_pass_history = []

    if diminish:
        # initilize for distance diagnostic
        init_weight = model.linear.weight.clone().detach() # record inital weight theta_0
        prev_weight = model.linear.weight.clone().detach() # memorize theta_n/q for every n=q^k until max_epochs/q
        k = 0

    if thresh_basis is None:
        thresh_basis = lambda x: 1e-4

    itr = itr
    weight = model.linear.weight.clone().detach()

    def func_val(hyper_params):
        return test_lsp(testDataLoader, model, loss_fn_val, hyper_params, device)
    
    def func_train(hyper_params):
        return test_lsp(testDataLoader, model, loss_fn_train, hyper_params, device)
    
    loss_val = []
    loss_train = []
    loss_train_reg = []
    
    for t in range(max_epochs):
        
        stochastic_obj_val = 0.0
        num_quadpts = 10
        num_pts = 2000
        if len(lam_min) == 1:
            stochastic_obj_val = gauss_legendre_integral(func_val, lam_min[0], lam_max[0], num_quadpts)
        elif len(lam_min) == 2:
            stochastic_obj_val = gauss_legendre_integral_2d(func_val, lam_min, lam_max, num_quadpts)
        elif len(lam_min) > 2:
            stochastic_obj_val = monte_carlo(func_val, lam_min, lam_max, num_pts, distribution, 88)
            stochastic_obj_train = monte_carlo(func_val, lam_min, lam_max, num_pts, distribution, 42)
            # stochastic_obj_train_reg = monte_carlo(func_train, lam_min, lam_max, num_pts, distribution, 42)
            loss_val.append(stochastic_obj_val)
            loss_train.append(stochastic_obj_train)
            # loss_train_reg.append(stochastic_obj_train_reg)
            print(f"--------approximate solution path for # function evaluation = {itr} complete--------")
            print(f"# gradient call: {t}\t average loss on training: {stochastic_obj_train}")
            print(f"# gradient call: {t}\t average loss on validation: {stochastic_obj_val}")
        # run one pass of dataset
        if (opt_method == 'SGD') or (opt_method == 'Adam'):
            new_grad, weight, itr = train_lsp(itr, weight, trainDataLoader, model, loss_fn_train, optimizer, 
                                            lam_min=lam_min, lam_max=lam_max, 
                                            weighted_avg=weighted_avg, step_size=step_size, 
                                            const=const, distribution=distribution, alpha_beta=alpha_beta, device=device)
        elif opt_method == 'lbfgs':
            new_grad, weight, itr = train_lsp_bfgs(itr, weight, trainDataLoader, model, loss_fn_train, optimizer, 
                                            lam_min=lam_min, lam_max=lam_max, 
                                            step_size=step_size, 
                                            const=const, distribution=distribution, device=device)



    return loss_train, loss_val, loss_train_reg, weight