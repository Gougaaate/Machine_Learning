import numpy as np
from sigmoid import sigmoid
from costFunction import h,costFunction


def costFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    # Initialize some useful values
    m,n = X.shape   # number of training examples and parameters
    theta = theta.reshape((n,1)) # due to the use of fmin_tnc

    J = 0.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta

    # =============================================================
    J = costFunction(theta,X,y)
    sum =0.0
    for j in range(1,n):
        sum = sum + theta[j]**2
    J = J + (Lambda/(2*m)) *sum
    
    # =============================================================
    
    return J
