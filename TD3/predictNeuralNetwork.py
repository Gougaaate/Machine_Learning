import numpy as np

from sigmoid import sigmoid


def predictNeuralNetwork(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """

    m, _ = X.shape
    num_labels, _ = Theta2.shape
    a2 = sigmoid(X @ Theta1.T)
    a2 = np.hstack((np.ones((m, 1)), a2))
    a3 = sigmoid(a2 @ Theta2.T)

    p = np.argmax(a3, axis=1) + 1

    return p
