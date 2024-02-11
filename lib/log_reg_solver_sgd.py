import numpy as np
import torch
from torch import nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "CPU"
)

# trace_frequency is measured in number of batches. -1 means don't print
def train_SGD(dataloader, model, loss_fn, optimizer, distribution='uniform', trace_frequency=-1):
    model.train()
    actv = nn.Sigmoid()
    for batch, (X_train, y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)
        
        rndm_lam = torch.tensor(0.5)
        # SGD picks random regulation parameter lambda
        if distribution == 'uniform':
            rndm_lam = torch.torch.distributions.Uniform(0, 1).sample()
        # print(f"random lam = {rndm_lam}")
        
        # Compute predicted y_hat
        theta = model(rndm_lam.cpu())
        pred = torch.mm(X_train, theta[1:].view(-1, 1))
        if model.intercept:
            const = torch.ones(len(X_train), 1).to(device)
            pred += torch.mm(const, theta[0].view(-1, 1))
        pred = actv(pred)
        # print(theta[0])
        
        loss = (1 - rndm_lam) * loss_fn(pred.view(-1, 1), y_train.view(-1, 1))
        loss += rndm_lam * 0.5 * theta.norm(p=2)**2
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # if (trace_frequency > 0) & (batch % trace_frequency == 0):
        #     loss, current = loss.item(), (batch + 1) * len(X_train)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Test function
def test_SGD(dataloader, model, loss_fn, lam):
    model.eval() #important
    actv = nn.Sigmoid()
    with torch.no_grad():  #makes sure we don't corrupt gradients and is faster
        for batch, (X_test, y_test) in enumerate(dataloader):
          X_test, y_test = X_test.to(device), y_test.to(device)
          
          # Compute prediction error
          theta = model(lam)
          pred = actv(torch.mm(X_test, theta[1:].view(-1, 1)) + theta[0].item())
          # print(f"prediction = {pred}")
          
          oos = (1 - lam) * loss_fn(pred.view(-1, 1), y_test.view(-1, 1))
          oos += lam * 0.5 * theta.norm(p=2)**2
          
    return oos.item()
