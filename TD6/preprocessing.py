import numpy as np
import matplotlib.pyplot as plt

from loadImages import loadImages
from selectFeatureVectors import selectFeatureVectors
from displayFeatures2d import displayFeatures2d
from displayFeatures3d import displayFeatures3d


def preprocessing():
    img73, img87 = loadImages()
    img73 = img73[60:, :]
    img87 = img87[60:, :]

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(img73)
    ax2.imshow(img87)
    featLearn = 0
    return featLearn, img73, img87
