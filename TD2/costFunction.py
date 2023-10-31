import numpy as np
from sigmoid import sigmoid

def h(x, theta):
    return sigmoid(np.transpose(x) @ theta)

def costFunction(theta, X, y):
    """ computes the cost of using theta as the
    parameter for logistic regression."""

    m, n = X.shape
    theta = theta.reshape((n, 1))  # due to the use of fmin_tnc
    J = 0.
    for i in range(m):
        J = J - y[i] * np.log(h(X[i], theta)) - (1 - y[i]) * np.log(1 - h(X[i], theta))

    return J/m
