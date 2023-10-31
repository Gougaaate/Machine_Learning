import numpy as np
from sigmoid import sigmoid

def h_lr(M,i):
    return sigmoid(M[i])


def lrCostFunction(theta, X, y, Lambda):
    """computes the cost of using
    theta as the parameter for regularized logistic regression.
    """

    # preambule
    m,n = X.shape # 5,4
    theta = theta.reshape((n,1)) # (4,1)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #
    # Hint: The computation of the cost function and gradients can be
    #       efficiently vectorized. For example, consider the computation
    #
    #           sigmoid(X @ theta) or np.dot(X, theta)
    #
    #       Each row of the resulting matrix will contain the value of the
    #       prediction for that example. You can make use of this to vectorize
    #       the cost function and gradient computations. 
    #
    M = X@theta
    J = 0.

    for i in range(m):
        J = J -y[i]*np.log(h_lr(M,i)) - (1-y[i])* np.log(1-h_lr(M,i))
    J = J/m
    
    ## 5.3.3 (Reg. part)
    sum =0.0
    for j in range(1,n):
        sum = sum + theta[j]**2
    J = J + (Lambda/(2*m)) *sum
    # =============================================================

    return J
