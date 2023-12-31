import numpy as np
from sigmoid import sigmoid
from costFunction import h

def dJ_j(X,y,theta,j,Lambda):
    m = X.shape[0] 
    dJ_j = 0.
    if j == 0:
        for i in range(m):
            dJ_j = dJ_j + (h(X[i],theta)-y[i])*X[i][j]

    else :
        for i in range(m):
            dJ_j = dJ_j + (h(X[i],theta)-y[i])*X[i][j]
        dJ_j = dJ_j + Lambda*theta[j]    
    dJ_j = dJ_j/m
    return dJ_j

def gradientFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    
    # Initialize some useful values
    m,n = X.shape   # number of training examples and parameters
    theta = theta.reshape((n,1)) # due to the use of fmin_tnc

    grad = 0.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of a particular choice of theta.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    # =============================================================
    save = np.zeros(n)

    for j in range(n):
        save[j] = dJ_j(X,y,theta,j,Lambda)
    grad = save

    # =============================================================

    return grad