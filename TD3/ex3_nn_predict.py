# %% Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

import numpy as np
import scipy.io  # Used to load the OCTAVE *.mat files
import scipy.misc  # Used to show matrix as an image
import matplotlib.pyplot as plt

from displayData import displayData
from predictNeuralNetwork import predictNeuralNetwork

# Set up the parameters you will use for this exercise
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10         # 10 labels, from 1 to 10
# (note that we have mapped "0" to label 10)

# %% =========== Part : Loading and Visualizing Data =============

# Load Training Data
print('\n -------------------------- \n')
print('Loading and Visualizing Data ...')
datafile = 'ex3data1.mat'
mat = scipy.io.loadmat(datafile)
X, y = mat['X'], mat['y']
m, n = X.shape

# Randomly select 100 data points to display
sel = np.random.permutation(range(m))
sel = sel[0:100]

displayData(X[sel, :])

# %% ================ : Loading Pameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('\n -------------------------- \n')
print('Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2

data = scipy.io.loadmat('ex3weights.mat')
Theta1 = data['Theta1']
Theta2 = data['Theta2']

# %% ================= Part : Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

# Add a column to X
X = np.column_stack((np.ones((m, 1)), X))

# prediction
pred = predictNeuralNetwork(Theta1, Theta2, X)

# evaluation
print('\n -------------------------- \n')
print('Training Set Accuracy: %f', np.mean(np.double(pred == np.squeeze(y))) * 100)
print('Expected training Set Accuracy: 97.5%')

#  To give you an idea of the network's output, you can also run
#  through the examples one at the time to see what it is predicting.

#  Randomly permute examples
rp = np.random.permutation(range(m))
X = X[:, 1:]
for i in range(m):
    plt.figure()
    X2 = np.array([X[rp[i], :]])
    displayData(X2)

    X2 = np.array(X[rp[i], :])
    X2 = np.concatenate(([1.], X2))
    pred = predictNeuralNetwork(Theta1, Theta2, X2.reshape((n + 1, 1)).T)
    pred = np.squeeze(pred)
    plt.title('Neural Network Prediction: %d (digit %d)\n' % (pred, np.mod(pred, 10)))

