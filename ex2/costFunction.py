import numpy as np
import sigmoid as s
def costFunction(theta,X,y):
	m=X.shape[0]
	h=s.sigmoid(X.dot(theta))
	cost=sum( -y*np.log(h) - (1.0-y)*np.log(1-h) )
	grad=X.T.dot(h-y)
	return ( cost/m , grad/m)

