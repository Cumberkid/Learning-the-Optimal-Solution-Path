import torch
from torch.optim.lr_scheduler import StepLR
from lib.lsp.basis_tf_module import Basis_TF_SGD
from lib.lsp.reg_solver_lsp import train_lsp
from lib.lsp.utils_lsp import get_sup_error_SGD

def learn_solution_path(input_dim, basis_dim, phi_lam, epochs, trainDataLoader, testDataLoader,
                        loss_fn, lam_min, lam_max, true_losses, lr=1e-3, alpha=1, init_lr=0.1,
                        diminish=False, gamma=0.1, dim_step=30, SGD=False, init_weight=None,
                        intercept=True, record_frequency=100, device='cpu', trace_frequency=-1):
    # build the model
    model = Basis_TF_SGD(input_dim, basis_dim, phi_lam, init_weight=init_weight, intercept=intercept).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    sup_err_history = []
    num_itr_history = []
    if diminish:
        # Define the learning rate scheduler
        scheduler = StepLR(optimizer, step_size=dim_step, gamma=gamma)  # Decrease LR by a factor of gamma every dim_step epochs

    for t in range(epochs):
        if SGD:
            # shrink learning rate
            lr = min([init_lr, alpha/(t+1)])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
        train_lsp(trainDataLoader, model, loss_fn, optimizer, device)
            
        if diminish:
            # Update the learning rate
            scheduler.step()
            
        if (t+1) % record_frequency == 0:
            num_itr_history.append(t+1)
            sup_err = get_sup_error_SGD(lam_min, lam_max, true_losses,
                                        model, testDataLoader, loss_fn, device)
            sup_err_history.append(sup_err)
            if (trace_frequency > 0) & ((t+1) % trace_frequency == 0):
                print(f"--------approximate solution path for # itr = {t+1} complete--------")
                print(f"# itr: {t+1}\t sup error: {sup_err}")

    return num_itr_history, sup_err_history, model
