import numpy as np

from sigmoid import sigmoid

def predictNeuralNetwork(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """

# Useful values
    m, _ = X.shape
    num_labels, _ = Theta2.shape
    p = np.arange(1, num_labels + 1)
#
# Hint: The max function might come in useful. In particular, the np.argmax
#       function can return the index of the max element, for more
#       information see 'numpy.argmax' on the numpy website. If your examples 
#       are in rows, then, you can use np.argmax(probs, axis=1) to obtain the 
#       max for each row.

    
    return p

