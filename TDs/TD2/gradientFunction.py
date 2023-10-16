from sigmoid import sigmoid
import numpy as np
from costFunction import h

def dJ_j(X,y,theta,j):
    m = X.shape[0] 
    dJ_j = 0.
    for i in range(m):
        dJ_j = dJ_j + (h(X[i],theta)-y[i])*X[i][j]
    dJ_j = dJ_j/m
    return dJ_j

def gradientFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic 
    regression and the gradient of the cost w.r.t. to the parameters.
    """

    # Initialize some useful values
    # number of training examples 
    m = X.shape[0]   

    # number of parameters
    n = X.shape[1]   
    theta = theta.reshape((n,1)) # due to the use of fmin_tnc


    # gradient variable
    grad = 0.
    save = np.zeros(n)
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of a particular choice of theta.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    for j in range(n):
        save[j] = dJ_j(X,y,theta,j)
    grad = save
    # =============================================================

    return grad