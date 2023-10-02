import numpy as np

def computeCostMulti(X, y, theta):  
    """
       computes the cost of using theta as the parameter for linear 
       regression to fit the data points in X and y
    """
    m = y.size
    J = (1 / (2 * m)) * np.sum(np.square(np.dot(X, theta)-y))

    return J
