import numpy as np
def computeCost(X,y,theta):
	d=np.matmul(X,theta)-y
	sq=np.square(d)
	s=np.sum(sq)	
	J=s/40
	return J
