def unif_reg_logit(lam, X, y, model):
    pred = model(X)
    loss = (1 - lam) * model.criterion(pred.view(-1, 1), y.view(-1, 1))
    loss = loss/len(X) + lam * 0.5 * model.ridge_term()
    return loss

def unif_weighted_logit(lam, X, y, model):
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

def reg_unif_weighted_logit(lam, X, y, model):
    loss = unif_weighted_logit(lam, X, y, model)
    loss += 0.25 * 0.5 * model.ridge_term()

    return loss

def exp_weighted_logit(lam, X, y, model):
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

def reg_exp_weighted_logit(lam, X, y, model):
    loss = exp_weighted_logit(lam, X, y, model)
    loss += 0.25 * 0.5 * model.ridge_term()

    return loss
