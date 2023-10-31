import numpy as np
from sigmoid import sigmoid
from costFunction import h


def predict(theta, X):
    """ computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    """

    p = 0
    m = X.shape[0]
    p = np.zeros(m)
    for i in range(m):
        if h(X[i], theta) >= 0.5:
            p[i] = 1

    return p
