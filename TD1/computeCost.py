import numpy as np


def h(x, theta):
    return theta[0] + theta[1] * x[1]
def computeCost(X, y, theta):
    """
       Computes the cost of using theta as the parameter for linear
       regression to fit the data points in X and y
    """

    m = y.size
    J = 0.
    for i in range(m):
        J = J + (h(X[i], theta) - y[i]) ** 2
    J = (1/(2*m)) * J

    return J
