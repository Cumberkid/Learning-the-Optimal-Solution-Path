import torch

# itr: input is number of iterations run before current epoch, returns number of iterations run after current epoch
# avg_weight: keeps track of and updates weighted average iterates including before current epoch according to Lacoste-Julien et al.
# step_size function take in 2 parameters: current iteration number, and a self-defined constant const
# step_size function returns the learning rate for current iteration
def train_lsp(itr, init_weight, dataloader, model, loss_fn, optimizer, weighted_avg=True, 
              step_size=None, const=None, distribution='uniform', device='cpu'):
    model.train()
    if weighted_avg:
        avg_weight = init_weight
    for batch, (X_train, y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)

        rndm_lam = torch.tensor(0.5)
        # SGD picks random regulation parameter lambda
        if distribution == 'uniform':
            rndm_lam = torch.torch.distributions.Uniform(0, 1).sample().cpu()
        elif distribution == 'exponential':
            rndm_lam = torch.torch.distributions.Exponential(0.1).sample().cpu()
            while rndm_lam > 20:
                randm_lam/=2

        loss = loss_fn(rndm_lam, X_train, y_train, model, device)

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
        
        if weighted_avg:
            # update weighted average iterates
            rho = 2 / (itr+3)
            itr += 1
            if itr > 50: # forgets first 50 iterations which are usually bad
                avg_weight = (1-rho) * avg_weight + rho * model.linear.weight.clone().detach()
            else:
                avg_weight = model.linear.weight.clone().detach()
    if weighted_avg:
        return grad, avg_weight, itr
    else:
        return grad, model.linear.weight.clone().detach(), itr
    
# Test function computes loss for a fixed input hyperparameter lam
def test_lsp(dataloader, model, loss_fn, lam, device='cpu'):
    model.eval() #important
    with torch.no_grad():  #makes sure we don't corrupt gradients and is faster
        for batch, (X_test, y_test) in enumerate(dataloader):
          X_test, y_test = X_test.to(device), y_test.to(device)
          
          oos = loss_fn(lam, X_test, y_test, model, device)
          
    return oos.item()
