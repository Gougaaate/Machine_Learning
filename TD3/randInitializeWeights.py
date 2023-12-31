import numpy as np

def randInitializeWeights(L_in, L_out):
    """randomly initializes the weights of a layer with L_in incoming connections and L_out outgoing
      connections.

      Note that W should be set to a matrix of size(L_out, 1 + L_in) as the 1st column of W handles the "bias" terms
    """
    epsilon = 0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon - epsilon
    return W
