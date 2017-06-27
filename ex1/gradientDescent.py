import numpy as np
import computeCost as cc
def gradientDescent(X,y,theta,alpha,iterations):
	for i in range(iterations):
		d=np.matmul(X,theta)-y
		theta=theta-alpha*(np.divide(np.matmul(np.transpose(X),d),20))
	return theta
