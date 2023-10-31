import numpy as np
from computeCost import computeCost, h


def f(x, y, theta, i):
    return (h(x, theta) - y) * x


def gradientDescent(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta, cost_history, theta_history = gradientDescent(X, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """
    m = y.size  # number of training examples
    n = theta.size  # number of parameters
    cost_history = np.zeros(num_iters)  # cost over iters
    theta_history = np.zeros((n, num_iters))  # theta over iters

    for i in range(num_iters):
        theta = theta - (alpha / m) * (X.T @ (X @ theta - y))
        cost_history[i] = computeCost(X, y, theta)
        theta_history[:, i] = theta.reshape((2,))

    #   ============================================================

    return theta, cost_history, theta_history
