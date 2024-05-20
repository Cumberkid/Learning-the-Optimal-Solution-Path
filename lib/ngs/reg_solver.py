import torch
"""The "train" function executes optimization on the input dataset w.r.t. the input loss function with the input optimizer on the ridge-regularized regression objective $h(\theta, \lambda) = (1-\lambda)BCE(X\theta, y) + \frac{\lambda}{2}\|\theta\|^2$. We will use the pytorch built-in SGD optimizer later, but note that this optimizer is actually just a deterministic gradient descent program.

To randomize for SGD, we notice that the loss function is a sum of losses of all training data points, and a standard SGD would randomly choose one of those points to descend on at each step of descent.

To speed up, we use a batch of data points to replace a single data point at each step of descent. When batch size = 1, this is equivalent to a standard SGD; and when batch size = training set size, this is simply a deterministic gradient descent.
"""

# trace_frequency is measured in number of batches. -1 means don't print
def train(itr, avg_weight, dataloader, model, loss_fn, optimizer, step_size=None, const=None, device='cpu'):
    # size = len(dataloader.dataset)
    model.train()
    avg_weight = avg_weight #initialize for weighted average
    # here, the "batch" notion takes care of randomization
    for batch, (X_train, y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)
        # print(batch, len(X_train))

        loss = loss_fn(model.reg_param, X_train, y_train, model)
        
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
        if itr>50:
            avg_weight = (1-rho) * avg_weight + rho * model.linear.weight.clone().detach()
        else:
            avg_weight = model.linear.weight.clone().detach()
            
    return itr, avg_weight

# test function computes objective loss for a specific input hyperparameter lam
def test(dataloader, model, loss_fn, lam, device='cpu'):
    model.eval() # important
    with torch.no_grad():  # makes sure we don't corrupt gradients and is faster
        for batch, (X_test, y_test) in enumerate(dataloader):
            X_test, y_test = X_test.to(device), y_test.to(device)
            
            # Compute objective loss
            oos = loss_fn(lam, X_test, y_test, model)
            
    return oos.item()
