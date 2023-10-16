from sigmoid import sigmoid
import numpy as np


def gradientFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic 
    regression and the gradient of the cost w.r.t. to the parameters.
    """

    # Number of training examples
    m, n = X.shape
    theta = theta.reshape((n, 1))  # due to the use of fmin_tnc

    grad = 0
    grad = (1 / m) * X.T @ (sigmoid(X @ theta) - y)

    return grad
