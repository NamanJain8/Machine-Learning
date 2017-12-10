import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib import cm
import math
from ex8 import estimateGaussian
from ex8 import selectThreshold

if __name__ == '__main__':
	print "Loading and visualising data..."
	data=loadmat('ex8data2.mat')
	X=data['X']
	Xval=data['Xval']
	yval=data['yval'].flatten()
	mu,sigma2=estimateGaussian(X)

	print "Estimating parameter epsilon...."
	p=1
	for i in range(Xval.shape[1]):
		p_i=(1.0/np.sqrt(2*math.pi*sigma2[i])) * np.exp( -(Xval[:,i]-mu[i])**2/(2*sigma2[i]) )
		p*=p_i

	epsilon,f1_score=selectThreshold(p,yval)

	p=1
	for i in range(X.shape[1]):
		p_i=(1.0/np.sqrt(2*math.pi*sigma2[i])) * np.exp( -(X[:,i]-mu[i])**2/(2*sigma2[i]) )
		p*=p_i
	print "Epsilon:  ",epsilon
	print "No. of anomalous examples in data set of 1000:  ",np.sum((p<epsilon)==1)