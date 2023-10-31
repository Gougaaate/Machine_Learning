import numpy as np


def linearRegCostFunction(X, y, theta, Lambda):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
    m, n = X.shape  # number of training examples
    theta = theta.reshape((n, 1))  # in case where theta is a vector (n,)

    J = (X @ theta - y).T @ (X @ theta - y) / (2 * m)
    J += (Lambda / (2 * m)) * np.sum(np.square(theta[1:]))
    grad = (1 / m) * X.T @ (X @ theta - y)
    grad[1:] += (Lambda / m) * theta[1:]
    return J.flatten(), grad.flatten()
