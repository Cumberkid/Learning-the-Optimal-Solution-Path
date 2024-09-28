import torch

# trace_frequency is measured in number of batches. -1 means don't print
def train(itr, init_weight, init_intercept, dataloader, model, loss_fn, optimizer, weighted_avg=False, step_size=None, const=None, device='cpu'):
    # size = len(dataloader.dataset)
    model.train()
    weight = init_weight #initialize
    intercept = init_intercept 
    # here, the "batch" notion takes care of randomization
    for batch, (X_train, y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)
        # print(batch, len(X_train))

        loss = loss_fn(model.hyper_param, X_train, y_train, model, device)
        
        if step_size is not None:
            # shrink learning rate as customized
            lr = step_size(itr, const)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update weighted average iterate
        rho = 2 / (itr+3)
        itr += 1
        if weighted_avg:
            if itr > 50:
                weight = (1-rho) * weight + rho * model.linear.weight.clone().detach().squeeze()
                if model.bias is not None:
                    intercept = (1-rho) * intercept + rho * model.linear.bias.clone().detach().squeeze()
            else:
                weight = model.linear.weight.clone().detach().squeeze()
                if model.bias is not None:
                    intercept = model.linear.bias.clone().detach().squeeze()
    return itr, weight, intercept

# test function computes objective loss for a specific input hyperparameter lam
def test(dataloader, model, loss_fn, hyper_params, device='cpu'):
    model.eval() # important
    with torch.no_grad():  # makes sure we don't corrupt gradients and is faster
        for batch, (X_test, y_test) in enumerate(dataloader):
            X_test, y_test = X_test.to(device), y_test.to(device)
            
            # Compute objective loss
            oos = loss_fn(hyper_params, X_test, y_test, model, device)
            
    return oos.item()
