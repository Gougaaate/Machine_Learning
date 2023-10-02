import numpy as np

def sigmoid(z):
    """computes the sigmoid of z."""
    g = np.array([[]])
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            g[i][j] = 1/(1+np.exp(z[i][j]))

    return g
