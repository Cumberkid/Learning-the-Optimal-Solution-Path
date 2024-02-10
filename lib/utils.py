import numpy as np
import torch

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, weights, lam, learning_rate, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        y_pred = sigmoid(np.dot(X, weights))
        gradient = (1-lam) * np.dot(X.T, y_pred - y) / m + lam * weights
        weights -= learning_rate * gradient
    return weights

def logit_by_hand(weight, intercept, lam, X_set, y_set):
    # weight = torch.zeros(input_dim, 1)
    pred = torch.sigmoid(torch.mm(X_set, weight) + intercept)
    # print(pred)
    criterion = torch.nn.BCELoss()
    soln = (1 - lam) * criterion(torch.squeeze(pred), y_set)
    soln += lam * 0.5 * (torch.squeeze(weight).norm(p=2)**2 + torch.squeeze(intercept)**2)

    return soln, pred
