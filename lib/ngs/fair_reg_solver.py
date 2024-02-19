import torch

"""The "fair_train" function is similar to the "train" function except that its objective function aims to treat two groups with fairness, i.e. $h(\theta, \lambda) = (1-\lambda)*loss(X_{\text{group A}}\theta, y_{\text{group A}}) + \lambda*loss(X_{\text{group B}}\theta, y_{\text{group B}})$.
"""

# trace_frequency is measured in number of batches. -1 means don't print
def fair_train(dataloader, model, loss_fn, optimizer, device="cpu", trace_frequency = -1):
    # size = len(dataloader.dataset)
    model.train()
    # here, the "batch" notion takes care of randomization
    for batch, (X_train, y_train) in enumerate(dataloader):
        # X_train, y_train = X_train.to(device), y_train.to(device)
        # print(batch, len(X_train))
        X_major = X_train[y_train == 1]
        y_major = torch.ones(len(X_major)).to(device)
        X_minor = X_train[y_train == 0]
        y_minor = torch.zeros(len(X_minor)).to(device)
        
        # Compute predicted y_hat
        pred_major = model(X_major)
        pred_minor = model(X_minor)
        
        # Fair loss function
        loss = (1 - model.reg_param) * loss_fn(pred_major.view(-1, 1), y_major.view(-1, 1))
        loss += model.reg_param * loss_fn(pred_minor.view(-1, 1), y_minor.view(-1, 1))
                
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # if (trace_frequency > 0) & (batch % trace_frequency == 0):
        #     loss, current = loss.item(), (batch + 1) * len(X_train)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

"""The "fair_test" function defined here is our objective function $h(\theta, \lambda) = (1-\lambda)*loss(X_{\text{group A}}\theta, y_{\text{group A}}) + \lambda*loss(X_{\text{group B}}\theta, y_{\text{group B}})$. The linear weight from the above trained model is our $\theta$."""

# Test function
def fair_test(dataloader, model, loss_fn, lam, device="cpu"):
    model.eval() # important
    with torch.no_grad():  # makes sure we don't corrupt gradients and is faster
        for batch, (X_test, y_test) in enumerate(dataloader):
            # X_test, y_test = X_test.to(device), y_test.to(device)
            
            X_major = X_test[y_test == 1]
            y_major = torch.ones(len(X_major)).to(device)
            X_minor = X_test[y_test == 0]
            y_minor = torch.zeros(len(X_minor)).to(device)
            
            # Compute predicted y_hat
            pred_major = model(X_major)
            pred_minor = model(X_minor)
            
            # With regularization
            oos = (1 - lam) * loss_fn(pred_major.view(-1, 1), y_major.view(-1, 1)) 
            oos += lam * loss_fn(pred_minor.view(-1, 1), y_minor.view(-1, 1))
                    
    return oos.item()
