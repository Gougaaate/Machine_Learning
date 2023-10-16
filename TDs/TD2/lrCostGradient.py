import numpy as np
from sigmoid import sigmoid


def Beta(X, theta, y):
    m = X.shape[0]
    M = X @ theta
    h_theta = sigmoid(M)

    return (h_theta - y)


def lrCostGradient(theta, X, y, Lambda):
    """computes the gradient of the cost  w.r.t. to the parameters 
    theta for regularized logistic regression .
    """

    # préambule
    m, n = X.shape  # m = 5; n = 4
    theta = theta.reshape((n, 1))  # (4,1)
    grad = 0.

    beta = Beta(X, theta, y)
    grad = (1 / m) * np.transpose(X) @ beta
    cpy = theta
    cpy[0] = 0

    grad = grad + (Lambda / m) * cpy

    return grad.flatten()  # ATTENTION: à conserver pour utiliser scipy.optimization.fmin_cg
