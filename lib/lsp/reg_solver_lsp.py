import torch

# itr: input is number of iterations run before current epoch, returns number of iterations run after current epoch
# avg_weight: keeps track of and updates weighted average iterates including before current epoch according to Lacoste-Julien et al.
def train_lsp(itr, t, avg_weight, dataloader, model, loss_fn, optimizer, step_size=None, const=None, distribution='uniform', device='cpu'):
    model.train()
    avg_weight = avg_weight
    for batch, (X_train, y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)

        rndm_lam = torch.tensor(0.5)
        # SGD picks random regulation parameter lambda
        if distribution == 'uniform':
            rndm_lam = torch.torch.distributions.Uniform(0, 1).sample().cpu()
        elif distribution == 'exponential':
            rndm_lam = torch.torch.distributions.Exponential(1).sample().cpu()

        loss = loss_fn(rndm_lam, X_train, y_train, model, device)

        if step_size is not None:
            # shrink learning rate as customized
            lr = step_size(t, const)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # record raw gradient
        grad = model.linear.weight.grad.clone().detach()
        optimizer.step()

        # update weighted average iterates
        rho = 2 / (itr+3)
        itr += 1
        if itr>50: # forgets first 50 iterations which are usually bad
            avg_weight = (1-rho) * avg_weight + rho * model.linear.weight.clone().detach()
        else:
            avg_weight = model.linear.weight.clone().detach()

    return grad, avg_weight, itr
    
# Test function computes loss for a fixed input hyperparameter lam
def test_lsp(dataloader, model, loss_fn, lam, device='cpu'):
    model.eval() #important
    with torch.no_grad():  #makes sure we don't corrupt gradients and is faster
        for batch, (X_test, y_test) in enumerate(dataloader):
          X_test, y_test = X_test.to(device), y_test.to(device)
          
          oos = loss_fn(lam, X_test, y_test, model, device)
          
    return oos.item()
