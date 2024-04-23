import torch

# trace_frequency is measured in number of batches. -1 means don't print
def train_lsp(dataloader, model, loss_fn, optimizer, distribution='uniform', device='cpu', trace_frequency=-1):
    model.train()

    for batch, (X_train, y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)
        
        rndm_lam = torch.tensor(0.5)
        # SGD picks random regulation parameter lambda
        if distribution == 'uniform':
            rndm_lam = torch.torch.distributions.Uniform(0, 1).sample().cpu()
        # print(f"random lam = {rndm_lam}")

        loss = loss_fn(rndm_lam, X_train, y_train, model, device)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        grad = model.linear.weight.grad.clone().detach()
        
        optimizer.step()
        
        # if (trace_frequency > 0) & (batch % trace_frequency == 0):
        #     loss, current = loss.item(), (batch + 1) * len(X_train)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return grad
    
# Test function
def test_lsp(dataloader, model, loss_fn, lam, device='cpu'):
    model.eval() #important
    with torch.no_grad():  #makes sure we don't corrupt gradients and is faster
        for batch, (X_test, y_test) in enumerate(dataloader):
          X_test, y_test = X_test.to(device), y_test.to(device)
          
          oos = loss_fn(lam, X_test, y_test, model, device)
          
    return oos.item()
