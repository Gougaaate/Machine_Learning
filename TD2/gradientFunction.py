import numpy as np
from costFunction import h

def dJ_j(X,y,theta,j):
    m = X.shape[0] 
    dJ_j = 0.
    for i in range(m):
        dJ_j = dJ_j + (h(X[i],theta)-y[i])*X[i][j]
    return dJ_j/m

def gradientFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression with regularization
    computes the cost of using theta as the parameter for regularized logistic 
    regression and the gradient of the cost w.r.t. to the parameters.
    """

    m = X.shape[0]
    n = X.shape[1]
    theta = theta.reshape((n,1)) # due to the use of fmin_tnc
    save = np.zeros(n)

    for j in range(n):
        save[j] = dJ_j(X,y,theta,j)
    grad = save

    return grad