import torch
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

"""The "train" function executes optimization on the input dataset w.r.t. the input loss function with the input optimizer on the ridge-regularized regression objective $h(\theta, \lambda) = (1-\lambda)BCE(X\theta, y) + \frac{\lambda}{2}\|\theta\|^2$. We will use the pytorch built-in SGD optimizer later, but note that this optimizer is actually just a deterministic gradient descent program.

To randomize for SGD, we notice that the loss function is a sum of losses of all training data points, and a standard SGD would randomly choose one of those points to descend on at each step of descent.

To speed up, we use a batch of data points to replace a single data point at each step of descent. When batch size = 1, this is equivalent to a standard SGD; and when batch size = training set size, this is simply a deterministic gradient descent.
"""

# trace_frequency is measured in number of batches. -1 means don't print
def train(dataloader, model, loss_fn, optimizer, trace_frequency = -1):
    # size = len(dataloader.dataset)
    model.train()
    # here, the "batch" notion takes care of randomization
    for batch, (X_train, y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)
        # print(batch, len(X_train))
        
        # Compute predicted y_hat
        pred = model(X_train)
        # With regularization
        loss = (1 - model.reg_param) * loss_fn(pred.view(-1, 1), y_train.view(-1, 1))
        loss += model.reg_param * 0.5 * model.ridge_term()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # if (trace_frequency > 0) & (batch % trace_frequency == 0):
        #     loss, current = loss.item(), (batch + 1) * len(X_train)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

"""The "test" function defined here is our objective function $h(\theta, \lambda) = (1-\lambda)BCE(X\theta, y) + \frac{\lambda}{2}\|\theta\|^2$. The linear weight from the above trained model is our $\theta$."""

# Test function
def test(dataloader, model, loss_fn, lam):
    model.eval() # important
    with torch.no_grad():  # makes sure we don't corrupt gradients and is faster
        for batch, (X_test, y_test) in enumerate(dataloader):
            X_test, y_test = X_test.to(device), y_test.to(device)
            
            # Compute prediction error
            pred = model(X_test)
            
            oos = (1 - lam) * loss_fn(pred.view(-1, 1), y_test.view(-1, 1))
            oos += lam * 0.5 * model.ridge_term()
            
    return oos.item()
