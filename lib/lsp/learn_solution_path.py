import torch
import math
from collections import deque
from lib.lsp.basis_tf_module import Basis_TF_SGD
from lib.lsp.reg_solver_lsp import train_lsp
from lib.lsp.utils_lsp import get_sup_error_lsp


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


# step_size function take in 2 parameters: current iteration number, and a self-defined constant
# step_size function returns the learning rate for current iteration
def learn_solution_path(input_dim, basis_dim, phi_lam, max_epochs, trainDataLoader, testDataLoader, loss_fn, lam_min, lam_max,
                        true_losses, init_lr=1e-3, diminish=False, gamma=0.97, q=1.3, k_0=5, thresh_lr=0.6, 
                        step_size=None, const=None, init_weight=None, intercept=True, weighted_avg=True, thresh_basis=1e-5,
                        record_frequency=10, distribution='uniform', device='cpu', trace_frequency=-1):
    # build the model
    model = Basis_TF_SGD(input_dim, basis_dim, phi_lam, init_weight=init_weight, intercept=intercept).to(device)
    lr=init_lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()

    # memorize the last 10 gradient info to decide if need to add more basis function
    norm_grad_list = deque(maxlen=10)

    # initialize weighted averaging sum
    if weighted_avg:
        avg_model = Basis_TF_SGD(input_dim, basis_dim, phi_lam, init_weight=init_weight, intercept=intercept).to(device)
        avg_weight = avg_model.linear.weight.clone().detach()
        sup_err_history_avg = []

    sup_err_history = []
    num_itr_history = []

    if diminish:
        # initilize for distance diagnostic
        init_weight = model.linear.weight.clone().detach() # record inital weight theta_0
        prev_weight = model.linear.weight.clone().detach() # memorize theta_n/q for every n=q^k until max_epochs/q
        k = 0

    for t in range(max_epochs):
        if step_size is not None:
            # shrink learning rate as customized
            lr = step_size(t, const)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # print(f"beta {const}/iteration #{t+1}, new lr = {lr}")

        # run one pass of dataset
        new_grad = train_lsp(trainDataLoader, model, loss_fn, optimizer, distribution, device)

        if diminish:
            # run distance diagnostic
            curr_weight = model.linear.weight.clone().detach()
            distance_diag, k, prev_weight = distance(init_weight, prev_weight, curr_weight, t+1, k, thresh_lr, q, k_0)
            if distance_diag:
                lr = gamma * lr
                # print(f"diminish at iteration #{t+1}, new lr = {lr}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        # update weighted average of iterates
        if weighted_avg and t>50:
            rho = 2 / (t+3)
            avg_weight = (1-rho) * avg_weight + rho * model.linear.weight.clone().detach()
        else:
            avg_weight = model.linear.weight.clone().detach()

        # record iteration result
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

        # when the change in second moment of gradient is small enough,
        # stop to add more basis functions
        new_grad_norm = new_grad.norm(p=2)**2
        if len(norm_grad_list) >= 10 and abs(norm_grad_list[0] - new_grad_norm) < thresh_basis:
            break
        else:
            norm_grad_list.append(new_grad_norm)

    if weighted_avg:
        return num_itr_history, sup_err_history, avg_weight, lr
    else:
        return num_itr_history, sup_err_history, model.linear.weight.clone().detach(), lr


def lsp_boosting(input_dim, start_basis_dim, end_basis_dim, phi_lam, max_epochs, 
                 trainDataLoader, testDataLoader, loss_fn, lam_min, lam_max, true_losses, 
                 init_lr=1e-3, diminish=True, gamma=0.97, q=1.3, k_0=5, thresh_lr=0.6, 
                 step_size=None, const=None, init_weight=None, intercept=True, weighted_avg=False, 
                 thresh_basis=1e-5, record_frequency=10, distribution='uniform', device='cpu', trace_frequency=-1):
    if step_size is not None:
        diminish = False

    num_itr_history = []
    sup_err_history = []
    breaks = []
    weight = init_weight
    lr = init_lr
    break_itr = 0
    const = const

    for basis_dim in range(start_basis_dim, end_basis_dim):

        print(f"********** now running lsp with #basis dimension = {basis_dim} ***********")

        num_itr_current, sup_err_current, weight, lr = learn_solution_path(input_dim, basis_dim, phi_lam, max_epochs, trainDataLoader, testDataLoader, loss_fn, lam_min, lam_max,
                            true_losses, init_lr=lr, diminish=diminish, gamma=gamma, q=q, k_0=k_0, thresh_lr=thresh_lr,step_size=step_size, const=const, init_weight=weight,
                            intercept=intercept, weighted_avg=weighted_avg, thresh_basis=thresh_basis, record_frequency=record_frequency, distribution=distribution, device=device, trace_frequency=trace_frequency)

        num_itr_history += [(x + break_itr) for x in num_itr_current]
        sup_err_history += sup_err_current
        break_itr = num_itr_history[len(num_itr_history)-1]
        breaks.append(break_itr)
        if const is not None:
            const = const/(2**0.5)


        # increase basis dimension by 1
        weight = torch.cat((weight, torch.zeros(weight.size(dim=0), 1)), 1)

    return num_itr_history, sup_err_history, breaks
