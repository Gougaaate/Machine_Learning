# %% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

import numpy as np
import scipy.io  # Used to load the OCTAVE *.mat files
import scipy.misc  # Used to show matrix as an image
import matplotlib.pyplot as plt

from displayData import displayData
from lrCostFunction import lrCostFunction
from lrCostGradient import lrCostGradient
from learnOneVsAll import learnOneVsAll
from predictOneVsAll import predictOneVsAll

input_layer_size = 400  # 20x20 Input Images of Digits
num_labels = 10  # 10 labels, from 1 to 10
# (note that we have mapped "0" to label 10)

# %% =========== Part 1: Loading and Visualizing Data =============


# Load Training Data
print('\n -------------------------- \n')
print('Loading and Visualizing Data ...')
datafile = 'ex2data3.mat'
mat = scipy.io.loadmat(datafile)
X, y = mat['X'], mat['y']
m, n = X.shape

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[0:100], :]

displayData(sel)

# %% ============ Part 2: Vectorize Logistic Regression ============

# Verification for the vectorized version of cost and gradient
theta_t = np.array([[-2, -1, 1, 2]]).T
X_t = np.arange(1, 16, 1).reshape(3, 5).T / 10
m = X_t.shape[0]
X_t = np.column_stack((np.ones((m, 1)), X_t))

y_t = np.array([[1, 0, 1, 0, 1]]).T
lambda_t = 0;

J = lrCostFunction(theta_t, X_t, y_t, lambda_t)
grad = lrCostGradient(theta_t, X_t, y_t, lambda_t)

print('\n -------------------------- \n')
print('Cost: %f' % J)
print('Expected cost: 0.734819')
print('Gradients:' + str(grad))
print('Expected gradients: 0.14656137  0.05144159  0.12472227  0.19800296')

## Prediction

# Add ones to the X data matrix
m = X.shape[0]
X = np.column_stack((np.ones((m, 1)), X))

print('\n -------------------------- \n')
print('Training One-vs-All Logistic Regression...')

Lambda = 0.1
all_theta = learnOneVsAll(X, y, num_labels, Lambda)

# %% ================ Part 3: Predict for One-Vs-All ================
#  After ...
pred = predictOneVsAll(all_theta, X)

accuracy = np.mean(np.double(pred == np.squeeze(y))) * 100
print('\n -------------------------- \n')
print('Training Set Accuracy: %f\n' % accuracy)
print('Expected approx accuracy: 96.46%')

fig = plt.figure()
plt.plot(np.arange(1, 5001), y, 'ro', markersize=10)
plt.plot(np.arange(1, 5001), pred, 'bx', markersize=10)
