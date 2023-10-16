import numpy as np
from sigmoid import sigmoid

def h(x,theta):
    return sigmoid(np.transpose(x)@theta)

def costFunction(theta, X, y):
    """ computes the cost of using theta as the
    parameter for logistic regression."""

	# Initialize some useful values
    m,n = X.shape   # number of training examples and parameters
    theta = theta.reshape((n,1)) # due to the use of fmin_tnc

    J = 0.
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #
    for i in range(m):
        J = J -y[i]*np.log(h(X[i],theta)) - (1-y[i])* np.log(1-h(X[i],theta))

    J = J/m
        
    # =============================================================
    
    return J

