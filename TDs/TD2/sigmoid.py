import numpy as np
def sigmoid(x):
    """
    Compute the sigmoid of an array
    """
    return 1 / (1 + np.exp(-x))
