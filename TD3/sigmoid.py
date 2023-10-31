import numpy as np


def sigmoid(z):
    """computes the sigmoid of z."""

    return 1 / (1 + np.exp(-z))
