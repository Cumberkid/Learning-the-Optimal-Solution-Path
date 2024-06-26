import torch

def unif_reg_logit(lam, X, y, model, device='cpu'):
    # Compute predicted y_hat
    theta = model(lam, device)
    
    if model.intercept:
        pred = torch.mm(X, theta[1:].view(-1, 1))
        const = torch.ones(len(X), 1).to(device)
        pred += torch.mm(const, theta[0].view(-1, 1))
    else:
        pred = torch.mm(X, theta.view(-1, 1))
    # print(theta[0])
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    loss = (1 - lam) * criterion(pred.view(-1, 1), y.view(-1, 1))
    loss += lam * 0.5 * theta.norm(p=2) ** 2

    return loss

def exp_reg_logit(lam, X, y, model, device='cpu'):
    # Compute predicted y_hat
    theta = model(lam, device)
    
    if model.intercept:
        pred = torch.mm(X, theta[1:].view(-1, 1))
        const = torch.ones(len(X), 1).to(device)
        pred += torch.mm(const, theta[0].view(-1, 1))
    else:
        pred = torch.mm(X, theta.view(-1, 1))
    # print(theta[0])
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    loss = 1/(1+lam) * criterion(pred.view(-1, 1), y.view(-1, 1))
    loss += lam/(1+lam) * 0.5 * theta.norm(p=2) ** 2

    return loss
    
def unif_weighted_logit(lam, X, y, model, device='cpu'):
    X_major = X[y == 1]
    y_major = y[y == 1]
    X_minor = X[y == 0]
    y_minor = y[y == 0]

    # compute predicted y_hat
    theta = model(lam, device)
    # print(theta[0])
    
    if model.intercept:
        pred_major = torch.mm(X_major, theta[1:].view(-1, 1))
        pred_minor = torch.mm(X_minor, theta[1:].view(-1, 1))
        const_major = torch.ones(len(X_major), 1).to(device)
        pred_major += torch.mm(const_major, theta[0].view(-1, 1))
        const_minor = torch.ones(len(X_minor), 1).to(device)
        pred_minor += torch.mm(const_minor, theta[0].view(-1, 1))
    else:
        pred_major = torch.mm(X_major, theta.view(-1, 1))
        pred_minor = torch.mm(X_minor, theta.view(-1, 1))
        
    # reweighted loss function with bias correction due to mini-batching
    # full data size = 1000
    # positive = 956
    # negative = 44
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    loss = (1 - lam) * (1000/956) * criterion(pred_major.view(-1, 1), y_major.view(-1, 1))
    loss += lam * (1000/44) * criterion(pred_minor.view(-1, 1), y_minor.view(-1, 1))
       
    loss = loss/len(X)

    return loss
    
def exp_weighted_logit(lam, X, y, model, device='cpu'):
    X_major = X[y == 1]
    y_major = y[y == 1]
    X_minor = X[y == 0]
    y_minor = y[y == 0]

    # compute predicted y_hat
    theta = model(lam, device)
    # print(theta[0])
    
    if model.intercept:
        pred_major = torch.mm(X_major, theta[1:].view(-1, 1))
        pred_minor = torch.mm(X_minor, theta[1:].view(-1, 1))
        const_major = torch.ones(len(X_major), 1).to(device)
        pred_major += torch.mm(const_major, theta[0].view(-1, 1))
        const_minor = torch.ones(len(X_minor), 1).to(device)
        pred_minor += torch.mm(const_minor, theta[0].view(-1, 1))
    else:
        pred_major = torch.mm(X_major, theta.view(-1, 1))
        pred_minor = torch.mm(X_minor, theta.view(-1, 1))
        
    # reweighted loss function with bias correction due to mini-batching
    # full data size = 1000
    # positive = 956
    # negative = 44
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    loss = 1/(1+lam) * (1000/956) * criterion(pred_major.view(-1, 1), y_major.view(-1, 1))
    loss += lam/(1+lam) * (1000/44) * criterion(pred_minor.view(-1, 1), y_minor.view(-1, 1))
        
    loss = loss/len(X)

    return loss

def reg_unif_weighted_logit(lam, X, y, model, device="cpu"):
    X_major = X[y == 1]
    y_major = y[y == 1]
    X_minor = X[y == 0]
    y_minor = y[y == 0]

    # compute predicted y_hat
    theta = model(lam, device)
    # print(theta[0])
    
    if model.intercept:
        pred_major = torch.mm(X_major, theta[1:].view(-1, 1))
        pred_minor = torch.mm(X_minor, theta[1:].view(-1, 1))
        const_major = torch.ones(len(X_major), 1).to(device)
        pred_major += torch.mm(const_major, theta[0].view(-1, 1))
        const_minor = torch.ones(len(X_minor), 1).to(device)
        pred_minor += torch.mm(const_minor, theta[0].view(-1, 1))
    else:
        pred_major = torch.mm(X_major, theta.view(-1, 1))
        pred_minor = torch.mm(X_minor, theta.view(-1, 1))
        
    # reweighted loss function with bias correction due to mini-batching
    # full data size = 1000
    # positive = 956
    # negative = 44
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    loss = (1 - lam) * (1000/956) * criterion(pred_major.view(-1, 1), y_major.view(-1, 1))
    loss += lam * (1000/44) * criterion(pred_minor.view(-1, 1), y_minor.view(-1, 1))
    loss = loss/len(X) + 0.25 * 0.5 * theta.norm(p=2) ** 2

    return loss
    
def reg_exp_weighted_logit(lam, X, y, model, device="cpu"):
    X_major = X[y == 1]
    y_major = y[y == 1]
    X_minor = X[y == 0]
    y_minor = y[y == 0]

    # compute predicted y_hat
    theta = model(lam, device)
    # print(theta[0])
    
    if model.intercept:
        pred_major = torch.mm(X_major, theta[1:].view(-1, 1))
        pred_minor = torch.mm(X_minor, theta[1:].view(-1, 1))
        const_major = torch.ones(len(X_major), 1).to(device)
        pred_major += torch.mm(const_major, theta[0].view(-1, 1))
        const_minor = torch.ones(len(X_minor), 1).to(device)
        pred_minor += torch.mm(const_minor, theta[0].view(-1, 1))
    else:
        pred_major = torch.mm(X_major, theta.view(-1, 1))
        pred_minor = torch.mm(X_minor, theta.view(-1, 1))
        
    # reweighted loss function with bias correction due to mini-batching
    # full data size = 1000
    # positive = 956
    # negative = 44
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    loss = 1/(1+lam) *(1000/956) * criterion(pred_major.view(-1, 1), y_major.view(-1, 1))
    loss += lam/(1+lam) * (1000/44) * criterion(pred_minor.view(-1, 1), y_minor.view(-1, 1))
    loss = loss/len(X) + 0.25 * 0.5 * theta.norm(p=2) ** 2

    return loss
