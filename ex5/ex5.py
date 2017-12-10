import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
def displayData(X,y,show=False):
	plt.xlabel("Change in water level(x)")
	plt.ylabel("Water flowing out of the dam")
	plt.plot(X,y,'rx')
	if show==True:
		plt.show()
	return

def linearRegCostFunction(theta,X,y,lambda_):
	m,n=X.shape
	h=X.dot(theta)
	J=(0.5/m)*np.sum((h-y)**2) + (lambda_*0.5/m) * np.sum(theta[1:]**2)
	return J

def linearRegGradFunction(theta,X,y,lambda_):
	m,n=X.shape
	theta=theta.reshape(-1,1)
	h=X.dot(theta)
	grad = (1.0/m)*(X.T.dot(h-y))+ (lambda_/m)*np.r_[[[0]],theta[1:]]
	return grad.flatten()

def trainLinearReg(theta,X,y,lambda_):
	theta=theta.flatten()
	result=minimize(linearRegCostFunction,theta,args=(X,y,lambda_),method=None,jac=linearRegGradFunction,options={'maxiter':1000000,'disp':False})
	theta=result.x
	cost=result.fun
	return cost,theta

def learningCurve(Xtrain,ytrain,Xcv,ycv,lambda_):
	m,n=Xtrain.shape
	initial_theta=np.ones((n,1))
	J_train=np.zeros((m,1))
	J_cv=np.zeros((m,1))	
	for i in range(m):
		cost,theta=trainLinearReg(initial_theta,Xtrain[:i+1,:],ytrain[:i+1,:],lambda_)
		J_train[i]=linearRegCostFunction(theta,Xtrain[:i+1,:],ytrain[:i+1,:],0)
		J_cv[i]=linearRegCostFunction(theta,Xcv,ycv,0)
	return J_train,J_cv

def featureNormalize(X):
	mu=np.mean(X,axis=0)
	sigma=np.sqrt(np.mean((X-mu)**2,axis=0))
	X_norm=(X-mu)/sigma
	return X_norm,mu,sigma

if __name__ == '__main__':
	data=loadmat('ex5data1.mat')
	X=data['X']
	y=data['y']
	Xcv=data['Xval']
	ycv=data['yval']
	Xtest=data['Xtest']
	ytest=data['ytest']

	print("visualising data...")
	displayData(X,y,True)

	raw_input("Press any key to apply regularised Linear Regression")
	lambda_=0.0
	X_pass=np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
	initial_theta=np.array([[1],[1]])
	cost=linearRegCostFunction(initial_theta,X_pass,y,lambda_)
	grad=linearRegGradFunction(initial_theta,X_pass,y,lambda_)
	print "Cost for theta = [1,1] is : ",cost,grad

	raw_input("Press any key to optimize Cost function...")
	cost,theta=trainLinearReg(initial_theta,X_pass,y,0)
	displayData(X,y)
	xplt=np.linspace(-40,50,50)
	yplt=theta[0]+theta[1]*xplt
	plt.plot(xplt,yplt)
	plt.show()
	
	raw_input("Press any key to plot Learning Curve...")
	Xcv_pass=np.concatenate((np.ones((Xcv.shape[0],1)),Xcv),axis=1)
	J_train,J_cv=learningCurve(X_pass,y,Xcv_pass,ycv,0)
	print "Visulising error vs. no.of training examples..."
	error_points=np.arange(1,X.shape[0]+1,1)
	plt.title("Learning Curve for Linear Regression")
	plt.xlabel("No. of training examples")
	plt.ylabel("Error")
	plt.plot(error_points,J_train,label="Training error")
	plt.plot(error_points,J_cv,label="Validation error")
	plt.legend()
	plt.show()

	print "Computing for more number of features"
	poly = PolynomialFeatures(degree=8)
	X_poly = poly.fit_transform(X_pass[:,1].reshape(-1,1))	#produces matrix with polynomial features havinga all powers less than degree 

	regr2 = LinearRegression()
	regr2.fit(X_poly, y)									#produces Linear regression weights for X_poly and y

	raw_input("Press any key to visualise new optimised regression")
	# plot range for x
	plot_x = np.linspace(-60,45)
	# using coefficients to calculate y
	plot_y = regr2.intercept_+ np.sum(regr2.coef_*poly.fit_transform(plot_x.reshape(-1,1)), axis=1)

	plt.plot(plot_x, plot_y, label='Scikit-learn LinearRegression')
	plt.scatter(X_pass[:,1], y, s=50, c='r', marker='x', linewidths=1)
	plt.xlabel('Change in water level (x)')
	plt.ylabel('Water flowing out of the dam (y)')
	plt.title('Polynomial regression degree 8')
	plt.legend(loc=4)
	plt.show()
