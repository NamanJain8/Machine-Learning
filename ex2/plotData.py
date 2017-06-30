import matplotlib.pyplot as plt
import numpy as np

def plotData(X,y,show=True,reg=False):
	pos=(y==1).nonzero()[0]
	neg=(y==0).nonzero()[0]
	plt.xlabel('Score 1')
	plt.ylabel('Score 2')
	if reg==True:
		plt.xlabel('Microchip Score 1')
		plt.ylabel('Microchip Score 2')
	plt.plot(X[pos,0],X[pos,1],'r+')
	plt.plot(X[neg,0],X[neg,1],'go')
	if show==True:
		plt.show()
	return

