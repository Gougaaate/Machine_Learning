import numpy as np
from matplotlib import pyplot as plt


def plotData(X,y):

	
	pos = X[(y==1).flatten(),:]
	neg = X[(y==0).flatten(),:]
	plt.plot(pos[:,0], pos[:,1], 'o', markersize=7, markeredgecolor='green', markerfacecolor='green')
	plt.plot(neg[:,0], neg[:,1], '+', markersize=7, markeredgecolor='red')
	plt.legend(['Admitted (y=1)', 'Not admitted (y=0)'], loc='lower left', fontsize='10',numpoints=1)
	plt.grid(True)
	plt.ylabel('Exam 2 score')
	plt.xlabel('Exam 1 score')


