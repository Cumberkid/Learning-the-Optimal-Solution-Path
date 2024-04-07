def reg_logit(lam, X, y, model):
    pred = model(X)
    loss = (1 - lam) * model.criterion(pred.view(-1, 1), y.view(-1, 1))
    loss += lam * 0.5 * model.ridge_term()
    return loss

def weighted_logit(lam, X, y, model):
    X_major = X[y == 1]
    y_major = y[y == 1]
    X_minor = X[y == 0]
    y_minor = y[y == 0]

    # Compute predicted y_hat
    pred_major = model(X_major)
    pred_minor = model(X_minor)

    # Fair loss function
    loss = (1 - lam) * model.criterion(pred_major.view(-1, 1), y_major.view(-1, 1))
    loss += lam * model.criterion(pred_minor.view(-1, 1), y_minor.view(-1, 1))

    return loss

def reg_weighted_logit(lam, X, y, model):
    loss = weighted_logit(lam, X, y, model)
    loss += 0.25 * 0.5 * model.ridge_term()

    return loss
