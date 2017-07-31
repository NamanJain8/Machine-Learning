import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.cm as cm
from scipy.optimize import minimize
def displayData(X):
	m,n=X.shape
	display_rows=int(np.sqrt(m))
	display_cols=int(m/display_rows)
	ex_height=int(np.sqrt(n))
	ex_width=int(n/ex_height)
	display_matrix=np.ones((display_rows*ex_height,display_cols*ex_width))
	for i in range(display_rows):
		for j in range(display_cols):
			index=i*display_cols+j
			image=X[index,:].reshape(ex_height,ex_width)
			display_matrix[i*ex_height:(i+1)*ex_height,j*ex_width:(j+1)*ex_width]=image
	plt.imshow(display_matrix.T,cm.Greys)
	plt.show()

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoidGradient(z):
	return sigmoid(z).dot(1-sigmoid(z))	

def randInitialiseWeights(in_L,out_L):
	eps=0.12
	theta=(np.random.rand(out_L,1+in_L)*(-2.0*eps)+np.random.rand(out_L,1+in_L)*(2*eps))/2.0
	return theta
def nnCostFunctionVectorised(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda_):
	boundary=(input_layer_size+1)*(hidden_layer_size)
	theta1=nn_params[:boundary].reshape((hidden_layer_size,input_layer_size+1))
	theta2=nn_params[boundary:].reshape((num_labels,hidden_layer_size+1))
	m=X.shape[0]

	a1=X
	a2=sigmoid(np.matmul(a1,theta1.T))
	a2=np.concatenate((np.ones((m,1)),a2),axis=1)
	h=a3=sigmoid(np.matmul(a2,theta2.T))

	y=np.reshape(y,(np.shape(y)[0],1))
	y_vec=1*(np.arange(1,num_labels+1)==y)
	J=np.sum(-y_vec*np.log(h)-(1.0-y_vec)*np.log(1.0-h))
	J/=m
	J+=(lambda_/(2.0*m))*(np.sum(theta1[:,1:]**2)+np.sum(theta2[:,1:]**2))
	
	d3=a3-y_vec
	d2=np.dot(Theta2[:,1:],d3.T)*sigmoidGradient(np.matmul(a1,theta1.T)).T
	Delta1=np.dot(d2,a1)
	Delta2=np.dot(d3.T,a2)
	
	Theta1_reg = lambda_/m*Theta1
	Theta2_reg = lambda_/m*Theta2
	Theta1_reg[:,0] = 0
	Theta2_reg[:,0] = 0
	Theta1_grad = 1/m*Delta1 + Theta1_reg
	Theta2_grad = 1/m*Delta2 + Theta2_reg
	gradient = np.concatenate((Theta1_grad.flatten('C'), Theta2_grad.flatten('C')))
	return J,gradient

def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda_):
	boundary=(input_layer_size+1)*(hidden_layer_size)
	theta1=nn_params[:boundary].reshape((hidden_layer_size,input_layer_size+1))
	theta2=nn_params[boundary:].reshape((num_labels,hidden_layer_size+1))
	m=X.shape[0]
	possible_labels=np.arange(1,num_labels+1)
	theta1_grad=np.zeros_like(theta1)
	theta2_grad=np.zeros_like(theta2)
	cost=0.0
	for i in range(m):
		#forward stepping
		a1=X[i,:]
		a2=sigmoid(theta1.dot(a1))
		a2=np.concatenate((np.ones(1),a2))
		h=a3=sigmoid(theta2.dot(a2))
		#cost calculation
		y_vec=np.vectorize(int)(possible_labels==y[i])
		cost+=np.sum(-y_vec*np.log(h)-(1.0-y_vec)*np.log(1.0-h))
		#back_propagation
		delta3=a3-y_vec
		theta2_grad+=np.outer(delta3,a2)
		delta2=theta2.T.dot(delta3)*a2*(1.0-a2)
		theta1_grad+=np.outer(delta2[1:],a1)
	cost/=m
	theta1_grad/=m
	theta2_grad/=m
	#regularisation
	cost_reg=(lambda_/(2.0*m))*(np.sum(theta1[:,1:]**2)+np.sum(theta2[:,1:]**2))
	theta1_grad+=(lambda_/m)*np.concatenate((np.zeros((theta1.shape[0],1)),theta1[:,1:]),axis=1)
	theta2_grad+=(lambda_/m)*np.concatenate((np.zeros((theta2.shape[0],1)),theta2[:,1:]),axis=1)
	gradient=np.concatenate((theta1_grad.flatten(),theta2_grad.flatten()))
	return cost+cost_reg,gradient

def predict(theta1,theta2,X):
	m=X.shape[0]
	a1=X
	a2=sigmoid(np.matmul(a1,theta1.T))
	h=a3=sigmoid(np.matmul(np.concatenate((np.ones((m,1)),a2),axis=1),theta2.T))
	return 1+np.argmax(h,axis=1)

if __name__=='__main__':
	print "Loading and visualising data...."
	data=loadmat('ex4data1.mat')
	X=data['X']
	y=data['y'].flatten()
	sel=np.random.permutation(X)[:100]
	displayData(sel)
	print "Loading Saved Neural Network parameters...."
	theta=loadmat('ex4weights.mat')
	theta1=theta['Theta1']
	theta2=theta['Theta2']
	m=X.shape[0]
	X=np.concatenate((np.ones((m,1)),X),axis=1)
	input_layer_size=theta1.shape[1]-1
	hidden_layer_size=theta2.shape[1]-1
	num_labels=theta2.shape[0]
	lambda_=1.0
	nn_params=np.concatenate((theta1.flatten(),theta2.flatten()))
	cost,grad=nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,0)
	print "Cost is: %f \n Gradient is:\n%s"%(cost,grad)
	cost,grad=nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,1)
	print "With Regularisation::\n Cost is: %f \n Gradient is:\n%s"%(cost,grad)
	# Check sigmoid gradient
	print "Now checking sigmoid gradient...."
	gradient=sigmoidGradient(np.array([1.0,0.5,0.0,-0.5,-1.0]))
	print "Sigmoid Gradient evaluated at [1,.5,0,-.5,-1] is: %s"%gradient
	boundary=(input_layer_size+1)*(hidden_layer_size)
	initial_theta1=randInitialiseWeights(input_layer_size,hidden_layer_size)	
	initial_theta2=randInitialiseWeights(hidden_layer_size,num_labels)
	theta1=nn_params[:boundary].reshape((hidden_layer_size,input_layer_size+1))
	theta2=nn_params[boundary:].reshape((num_labels,hidden_layer_size+1))
	print "Training Neural Network...."
	initial_theta=np.concatenate((initial_theta1.flatten(),initial_theta2.flatten()))
	result=minimize(nnCostFunction,initial_theta,args=(input_layer_size,hidden_layer_size,num_labels,X,y,lambda_),method='CG',jac=True,options={'maxiter':50,'disp':False})
	nn_params=result.x	
	boundary=(input_layer_size+1)*(hidden_layer_size)
	theta1=nn_params[:boundary].reshape((hidden_layer_size,input_layer_size+1))
	theta2=nn_params[boundary:].reshape((num_labels,hidden_layer_size+1))
	print "Visualising Neural Networks...."
	# Display data
	predictions=predict(theta1,theta2,X)
	accuracy=100*np.mean(predictions==y)
	print "Training set accuracy: %0.2f%%"%accuracy




