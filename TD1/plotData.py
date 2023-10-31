import matplotlib.pyplot as plt
import numpy as np

def plotData(X, y):
    """
    plots the data points and gives the figure axes labels of
    population and profit.
    """
    fig = plt.figure()  # open a new figure window
    plt.plot(X, y, 'r+', markersize=10)
    plt.grid(True)  # Always plot.grid true!
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10.000s')
    return 0
