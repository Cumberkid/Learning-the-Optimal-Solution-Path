import torch
from torch import nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# majority class is set to be class 1
# trace_frequency is measured in number of batches. -1 means don't print
def fair_train_SGD(dataloader, model, loss_fn, optimizer, distribution='uniform', trace_frequency=-1):
    model.train()
    actv = nn.Sigmoid()
    for batch, (X_train, y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)
        
        X_major = X_train[y_train == 1]
        y_major = torch.ones(len(X_major)).to(device)
        X_minor = X_train[y_train == 0]
        y_minor = torch.zeros(len(X_minor)).to(device)
        
        rndm_lam = torch.tensor(0.5)
        # SGD picks random regulation parameter lambda
        if distribution == 'uniform':
            rndm_lam = torch.torch.distributions.Uniform(0, 1).sample()
        # print(f"random lam = {rndm_lam}")

        # compute predicted y_hat
        theta = model(rndm_lam.cpu())
        # print(theta[0])
        pred_major = torch.mm(X_major, theta[1:].view(-1, 1))
        pred_minor = torch.mm(X_minor, theta[1:].view(-1, 1))
        if model.intercept:
            const_major = torch.ones(len(X_major), 1).to(device)
            pred_major += torch.mm(const_major, theta[0].view(-1, 1))
            const_minor = torch.ones(len(X_minor), 1).to(device)
            pred_minor += torch.mm(const_minor, theta[0].view(-1, 1))
        pred_major = actv(pred_major)
        pred_minor = actv(pred_minor)
        # fair loss function
        loss = (1 - rndm_lam) * loss_fn(pred_major.view(-1, 1), y_major.view(-1, 1)) 
        loss += rndm_lam * loss_fn(pred_minor.view(-1, 1), y_minor.view(-1, 1))
                
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # if (trace_frequency > 0) & (batch % trace_frequency == 0):
        #     loss, current = loss.item(), (batch + 1) * len(X_train)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# test function for fair objective
def fair_test_SGD(dataloader, model, loss_fn, lam):
    model.eval() #important
    actv = nn.Sigmoid()
    with torch.no_grad():  #makes sure we don't corrupt gradients and is faster
        for batch, (X_test, y_test) in enumerate(dataloader):
            X_test, y_test = X_test.to(device), y_test.to(device)
            
            X_major = X_test[y_test == 1]
            y_major = torch.ones(len(X_major)).to(device)
            X_minor = X_test[y_test == 0]
            y_minor = torch.zeros(len(X_minor)).to(device)
            
            # compute prediction error
            theta = model(lam)
            pred_major = actv(torch.mm(X_major, theta[1:].view(-1, 1)) + theta[0].item())
            pred_minor = actv(torch.mm(X_minor, theta[1:].view(-1, 1)) + theta[0].item())
            # print(f"prediction = {pred}")
            
            oos = (1 - lam) * loss_fn(pred_major.view(-1, 1), y_major.view(-1, 1)) 
            oos += lam * loss_fn(pred_minor.view(-1, 1), y_minor.view(-1, 1))
                    
    return oos.item()
