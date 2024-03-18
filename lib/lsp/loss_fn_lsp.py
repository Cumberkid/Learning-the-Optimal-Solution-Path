import torch

criterion = torch.nn.BCEWithLogitsLoss()

def reg_logit(lam, X, y, model, device="cpu"):
    # Compute predicted y_hat
    theta = model(lam, device)
    pred = torch.mm(X, theta[1:].view(-1, 1))
    if model.intercept:
        const = torch.ones(len(X), 1).to(device)
        pred += torch.mm(const, theta[0].view(-1, 1))
    # print(theta[0])

    loss = (1 - lam) * criterion(pred.view(-1, 1), y.view(-1, 1))
    loss += lam * 0.5 * theta.norm(p=2) ** 2

    return loss

def weighted_logit(lam, X, y, model, device="cpu"):
    X_major = X[y == 1]
    y_major = y[y == 1]
    X_minor = X[y == 0]
    y_minor = y[y == 0]

    # compute predicted y_hat
    theta = model(lam, device)
    # print(theta[0])
    pred_major = torch.mm(X_major, theta[1:].view(-1, 1))
    pred_minor = torch.mm(X_minor, theta[1:].view(-1, 1))
    if model.intercept:
        const_major = torch.ones(len(X_major), 1).to(device)
        pred_major += torch.mm(const_major, theta[0].view(-1, 1))
        const_minor = torch.ones(len(X_minor), 1).to(device)
        pred_minor += torch.mm(const_minor, theta[0].view(-1, 1))
    # fair loss function
    loss = (1 - lam) * criterion(pred_major.view(-1, 1), y_major.view(-1, 1))
    loss += lam * criterion(pred_minor.view(-1, 1), y_minor.view(-1, 1))

    return loss

def reg_weighted_logit(lam, X, y, model, device="cpu"):
    loss = weighted_logit(lam, X, y, model, device)
    loss += 0.5 * theta.norm(p=2) ** 2

    return loss
