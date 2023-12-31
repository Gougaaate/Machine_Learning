#%% package
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


#%% def loadImages
def loadImages():
    ''' fonction de lecture des images 
    de la mer d'Aral 1973 et 1987
    '''

    # definition du chemin aux images
    path=''
    im73_filename = 'Aral1973_Clean.jpg';
    im87_filename = 'Aral1987_Clean.jpg';

    im73 = plt.imread(im73_filename)
    im87 = plt.imread(im87_filename)

    return im73, im87