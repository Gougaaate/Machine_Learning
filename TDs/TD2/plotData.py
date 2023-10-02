import numpy as np
from matplotlib import pyplot as plt


def plotData(X,y):
    """
    Plot data X with different markers according the value in y
    """
    n = X.shape[1]
    X_pos = X[(y == 1).flatten(), :]
    X_neg = X[(y == 0).flatten(), :]
    plt.plot(X_pos[:,0], X_pos[:,1], 'o', markeredgecolor='black', color='lightgreen')
    plt.plot(X_neg[:,0], X_neg[:,1], '+', color='red')

    plt.grid()
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    plt.show()