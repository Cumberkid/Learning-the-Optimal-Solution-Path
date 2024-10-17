import torch

def unif_reg_logit(lam, X, y, model, device='cpu'):
    # Transform the lam to [-1, 1] interval
    lam_transformed = 2 * lam - 1
    # Compute predicted y_hat
    theta = model(lam_transformed, device)
    
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

    # Transform the lam to [-1, 1] interval
    lam_transformed = 2 * lam - 1
    # compute predicted y_hat
    theta = model(lam_transformed, device)
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

    # Transform the lam to [-1, 1] interval
    lam_transformed = 2 * lam - 1
    # compute predicted y_hat
    theta = model(lam_transformed, device)
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

# balance expected return, risk, and diversification
# constraint is that sum of model weight should be 1. We achieve this by setting the first term of weight
# to equal (1- sum of remaining terms of weight)
def allocation_cost_no_con(hyper_params, decomp_cov, mean, model, device="cpu"):
    # compute predicted y_hat
    n = decomp_cov.shape[1]
    # Transform the lam to [-1, 1] interval
    lam_transformed = [2 * hyper_params[0] - 1, 2.5*hyper_params[1] - 1.5]
    theta = model(lam_transformed, device)
    # first_column = decomp_cov[:, 0].unsqueeze(1)
    risk = torch.mm(decomp_cov, theta.view(-1, 1))
    exp_rtrn = torch.mm(mean.view(-1, 1).T, theta.view(-1, 1))
    # a proximity smoothing on 1-norm    
    mu = .01
    cost = torch.sum(torch.sqrt(theta**2 + mu**2) - mu)

    # input hyperparameter lam is a 2-d array
    loss = hyper_params[0] * risk.norm(p=2)**2 - hyper_params[1] * exp_rtrn + cost

    return loss

# balance expected return, risk, and diversification
# constraint is that sum of model weight should be 1. We achieve this by setting the first term of weight
# to equal (1- sum of remaining terms of weight)
def allocation_cost(hyper_params, decomp_cov, mean, model, device="cpu"):
    # compute predicted y_hat
    n = decomp_cov.shape[1]
    # Transform the lam to [-1, 1] interval
    lam_transformed = [2 * hyper_params[0] - 1, 2.5*hyper_params[1] - 1.5]
    theta = model(lam_transformed, device)
    first_column = decomp_cov[:, 0].unsqueeze(1)
    risk = torch.mm(decomp_cov[:, 1:] - first_column.repeat(1, n - 1), theta.view(-1, 1)) + first_column
    exp_rtrn = torch.mm(mean[1:].view(-1, 1).T - mean[0].repeat(1, n - 1), theta.view(-1, 1)) + mean[0]
    # a proximity smoothing on 1-norm    
    mu = .01
    cost = torch.sum(torch.sqrt(theta**2 + mu**2)) + torch.sqrt((1-sum(theta))**2 + mu**2) - n * mu

    # input hyperparameter lam is a 2-d array
    loss = hyper_params[0] * risk.norm(p=2)**2 - hyper_params[1] * exp_rtrn + cost

    return loss