import torch

# itr: input is number of iterations run before current epoch, returns number of iterations run after current epoch
# avg_weight: keeps track of and updates weighted average iterates including before current epoch according to Lacoste-Julien et al.
# step_size function take in 2 parameters: current iteration number, and a self-defined constant const
# step_size function returns the learning rate for current iteration
def train_lsp(itr, init_weight, dataloader, model, loss_fn, optimizer, lam_min=[0], lam_max=[1], weighted_avg=True, 
              step_size=None, const=None, distribution='uniform', device='cpu'):
    model.train()
    avg_weight = init_weight
    for batch, (X_train, y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)

        hyper_params = []
        for i in range(len(lam_min)):
            hyper_params.append(torch.tensor(0.5))
            # SGD picks random regulation parameter lambda
            if distribution == 'uniform':
                hyper_params[i] = torch.torch.distributions.Uniform(lam_min[i], lam_max[i]).sample().cpu()
            elif distribution == 'exponential':
                hyper_params[i] = torch.torch.distributions.Exponential(0.1).sample().cpu()
                while hyper_params[i] > 20:
                    hyper_params[i]/=2

        # if standardize_params: # all n+1 hyper params have to sum up to 1
        #     dummy = torch.torch.distributions.Uniform(0, 1).sample().cpu()
        #     hyper_params = [x / (sum(hyper_params) + dummy) for x in hyper_params]

        loss = loss_fn(hyper_params, X_train, y_train, model, device)

        if step_size is not None:
            # shrink learning rate as customized
            for param_group in optimizer.param_groups:
                param_group['lr'] = step_size(itr, const)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # record raw gradient
        grad = model.linear.weight.grad.clone().detach()
        optimizer.step()
        
        rho = 2 / (itr+3)
        itr += 1
        if weighted_avg and itr > 50:
            # update weighted average iterates
            avg_weight = (1-rho) * avg_weight + rho * model.linear.weight.clone().detach()
        else:
            avg_weight = model.linear.weight.clone().detach()

    return grad, avg_weight, itr
    
# Test function computes loss for a fixed input hyperparameter lam
def test_lsp(dataloader, model, loss_fn, hyper_params, device='cpu'):
    model.eval() #important
    with torch.no_grad():  #makes sure we don't corrupt gradients and is faster
        for batch, (X_test, y_test) in enumerate(dataloader):
          X_test, y_test = X_test.to(device), y_test.to(device)
          
          oos = loss_fn(hyper_params, X_test, y_test, model, device)
          
    return oos.item()
