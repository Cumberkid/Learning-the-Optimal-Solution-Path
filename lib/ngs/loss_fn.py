import torch

def unif_reg_logit(lam, X, y, model, device="cpu"):
    pred = model(X)
    loss = (1 - lam) * model.criterion(pred.view(-1, 1), y.view(-1, 1))
    loss = loss/len(X) + lam * 0.5 * model.ridge_term()
    return loss

def unif_weighted_logit(lam, X, y, model, device="cpu"):
    X_major = X[y == 1]
    y_major = y[y == 1]
    X_minor = X[y == 0]
    y_minor = y[y == 0]

    # Compute predicted y_hat
    pred_major = model(X_major)
    pred_minor = model(X_minor)

    # reweighted loss function with bias correction due to mini-batching
    # full data size = 1000
    # positive = 956
    # negative = 44
    loss = (1 - lam) * (1000/956) * model.criterion(pred_major.view(-1, 1), y_major.view(-1, 1))
    loss += lam * (1000/44) * model.criterion(pred_minor.view(-1, 1), y_minor.view(-1, 1))
    loss = loss/len(X)

    return loss

def reg_unif_weighted_logit(lam, X, y, model, device="cpu"):
    loss = unif_weighted_logit(lam, X, y, model, device)
    loss += 0.25 * 0.5 * model.ridge_term()

    return loss

def exp_weighted_logit(lam, X, y, model, device="cpu"):
    X_major = X[y == 1]
    y_major = y[y == 1]
    X_minor = X[y == 0]
    y_minor = y[y == 0]

    # Compute predicted y_hat
    pred_major = model(X_major)
    pred_minor = model(X_minor)

    # reweighted loss function with bias correction due to mini-batching
    # full data size = 1000
    # positive = 956
    # negative = 44
    loss = 1/(1+lam) * (1000/956) * model.criterion(pred_major.view(-1, 1), y_major.view(-1, 1))
    loss += lam/(1+lam) * (1000/44) * model.criterion(pred_minor.view(-1, 1), y_minor.view(-1, 1))
    loss = loss/len(X)

    return loss

def reg_exp_weighted_logit(lam, X, y, model, device="cpu"):
    loss = exp_weighted_logit(lam, X, y, model)
    loss += 0.25 * 0.5 * model.ridge_term()

    return loss

# balance expected return, risk, and diversification
# constraint is that sum of model weight should be 1. We achieve this by setting the first term of weight
# to equal (1- sum of remaining terms of weight)
def allocation_cost_no_con(hyper_params, decomp_cov, mean, model, device='cpu'):
    n = decomp_cov.shape[1]
    risk = model(decomp_cov)
    exp_rtrn = model(mean.view(-1, 1).T)
    mu = .01
    theta = model(torch.eye(n).to(device))
    # a proximity smoothing on 1-norm    
    cost = torch.sum(torch.sqrt(theta**2 + mu**2) - mu)
    # input hyperparameter lam is a 2-d array
    loss = hyper_params[0] * risk.norm(p=2)**2 - hyper_params[1] * exp_rtrn + cost

    return loss

# balance expected return, risk, and diversification
# constraint is that sum of model weight should be 1. We achieve this by setting the first term of weight
# to equal (1- sum of remaining terms of weight)
def allocation_cost(hyper_params, decomp_cov, mean, model, device='cpu'):
    n = decomp_cov.shape[1]
    first_column = decomp_cov[:, 0].unsqueeze(1)
    risk = model(decomp_cov[:, 1:] - first_column.repeat(1, n - 1)) + first_column
    exp_rtrn = model(mean[1:].view(-1, 1).T - mean[0].repeat(1, n - 1)) + mean[0]
    mu = .01
    theta = model(torch.eye(n - 1).to(device))
    # a proximity smoothing on 1-norm    
    cost = torch.sum(torch.sqrt(theta**2 + mu**2)) + torch.sqrt((1-sum(theta))**2 + mu**2) - n * mu

    # input hyperparameter lam is a 2-d array
    loss = hyper_params[0] * risk.norm(p=2)**2 - hyper_params[1] * exp_rtrn + cost

    return loss
