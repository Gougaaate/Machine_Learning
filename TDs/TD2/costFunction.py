import numpy as np
from sigmoid import sigmoid


def costFunction(theta, X, y):
    """ computes the cost of using theta as the
    parameter for logistic regression."""

    m, n = X.shape  # Number of training examples and parameters
    theta = theta.reshape((n, 1))  # Due to the use of fmin_tnc

    cost1 = y.T @ np.log(sigmoid(X @ theta))
    cost2 = (1 - y).T @ np.log(1 - sigmoid(X @ theta))
    J = -1 / m * (cost2 + cost1)

    return J
