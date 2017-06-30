import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.cm as cm
from scipy.optimize import minimize

def display_data(X):
	m,n=X.shape
	display_rows=int(np.sqrt(m))
	display_cols=int(m/display_rows)
	example_rows=int(np.around(np.sqrt(n)))
	example_cols=int(n/example_rows)
	display_array=np.ones((display_rows*example_rows,display_cols*example_cols))
	for i in range(display_rows):
		for j in range(display_cols):
			index=i*display_cols+j
			image=X[index,:].reshape((example_rows,example_cols))
			display_array[i*example_rows:(i+1)*example_rows,j*example_cols:(j+1)*example_cols]=image

	plt.imshow(display_array.T,cm.Greys)
	plt.show()

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

	
def costFunction(theta,X,y):
	m=X.shape[0]
	h=sigmoid(X.dot(theta))
	cost=sum(-y*np.log(h)-(1.0-y)*np.log(1.0-h))
	grad=X.T.dot(h-y)
	return (cost/(2*m),grad/m)
	
def costFunctionReg(theta,X,y,lambda_):
	m=X.shape[0]
	cost,grad=costFunction(theta,X,y)
	reg_cost = (lambda_/(2.0*m))*np.sum(theta[1:]**2)
	reg_grad=(lambda_/m)*np.sum(theta)
	return cost+reg_cost,grad+reg_grad


def oneVsAll(X,y,cnt_labels,lambda_):
	n=X.shape[1]
	all_theta=np.zeros((cnt_labels,n))
	for i in range(1,cnt_labels+1):
		initial_theta=np.zeros(n)
		target=np.vectorize(int)(y==i)
		result=minimize(costFunctionReg,initial_theta,args=(X,target,lambda_),method='CG',jac=True,options={'maxiter':1000,'disp':False})
		theta=result.x
		cost=result.fun
		print "Training theta for label %d | cost: %f"%(i,cost)
		all_theta[i-1,:]=theta
	return all_theta

def predictOneVsAll(theta,X):
	return np.argmax(np.matmul(X,theta.T),axis=1)

if __name__=='__main__':
	data=loadmat('ex3data1.mat')
	X=data['X']
	y=data['y']
	y=y.flatten()
	sel=np.random.permutation(X)[:400]
	display_data(sel)

	m=X.shape[0]
	lambda_=1
	X=np.concatenate((np.ones((m,1)),X),axis=1)
	all_theta=oneVsAll(X,y,10,lambda_)
	predictions=predictOneVsAll(all_theta,X)
	accuracy=100*np.mean(predictions==y)
	print "Accuracy of training: %0.3f %%"%accuracy
