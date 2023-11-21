# %% Machine Learning Class - Exercise Aral Sea Surface Estimation

import numpy as np
import matplotlib.pyplot as plt

plt.ioff()  # to see figure avant input

from preprocessing import preprocessing

# %% Examen des données, prétraitements et extraction des descripteurs

featLearn, img73, img87 = preprocessing()
print(img87)
plt.show()

# %% Apprentissage / Learning / Training


# Apprentissage de la fonction de classement


# prediction des labels sur la base d'apprentissage


# Visualisation des resultats


# %% Classement et estimation de la diminution de surface
# Classifying / Predicting / Testing


# mise en forme de l'image de 1973 et 1987 en matrice Num Pixels / Val Pixels

# Classement des deux jeux de données et visualisation des résultats en image


# %% Estimation de la surface perdue
answer = input('Numero de la classe de la mer ? ')
cl_mer = int(answer)
