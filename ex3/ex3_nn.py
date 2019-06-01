import numpy as np
from scipy.io import loadmat
def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def predict(theta1,theta2,X):
	a1=X.T
	a2=sigmoid(np.matmul(theta1,a1))
	m=a2.shape[1]
	a2=np.concatenate((np.ones((1,m)),a2),axis=0)
	a3=sigmoid(np.matmul(theta2,a2))
	h=a3
	return 1+np.argmax(h,axis=0)
	


if __name__=='__main__':
	data=loadmat('ex3data1.mat')
	X=data['X']
	y=data['y'].flatten()
	m=X.shape[0]
	theta=loadmat('ex3weights.mat')
	theta1=theta['Theta1']
	theta2=theta['Theta2']
	X=np.concatenate((np.ones((m,1)),X),axis=1)
	predictions=predict(theta1,theta2,X)
	accuracy=100*np.mean(y==predictions)
	print "Accuracy of our hypothesis is: %0.2f%%"%accuracy
