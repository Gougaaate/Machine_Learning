import numpy as np
from computeCostMulti import computeCostMulti


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """
    # Initialize some useful values
    m = y.size  # number of training examples
    n = theta.size  # number of parameters
    cost_history = np.zeros(num_iters)
    theta_history = np.zeros((n, num_iters))

    for i in range(num_iters):
        theta = theta - (alpha / m) * np.dot(X.T, np.dot(X, theta) - y)

        cost_history[i] = computeCostMulti(X, y, theta)
        theta_history[:, i] = theta.reshape((n,))

    return theta, cost_history, theta_history