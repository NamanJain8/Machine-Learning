import numpy as np
from scipy.optimize import minimize
import costFunction as cf
import mapFeature as mf
import plotData as pd
import plotDecisionBound as pdb
import predict as p
import sigmoid as s
import fileinput




if __name__=='__main__':
	data1=np.loadtxt('ex2data1.txt',delimiter=',')
	X=data1[:,0:2]
	y=data1[:,2]
	pd.plotData(X,y)
	
	print "Data plotted"
	raw_input("Press Enter to move.....")

	m,n=X.shape
	X=np.concatenate((np.ones((m,1)),X), axis=1)
	initial_theta=np.zeros((n+1))
	cost,grad=cf.costFunction(initial_theta,X,y)
	print "Cost for initial theta: %f"%cost
	print "Gradient for initial theta: %s"%grad
	raw_input("Press Enter to get optimised value")

	objective = lambda t : cf.costFunction(t,X,y)[0]
	result=minimize( objective,initial_theta,method='Nelder-Mead',\
	options={'maxiter':1000,'disp':False,})
	
	theta=result.x
	cost=result.fun

	print "Cost at theta found: %f"%cost
	print "Theta found: %s"%theta
	raw_input("Plot Decision Bound")
	
	pdb.plotDecisionBound(theta,X,y)
	
	raw_input("Press enter to check accuracy of our data")
	predictions=p.predict(theta,X)
	accuracy=100*np.mean(predictions==y)
	print "Train Accuracy: %0.2f"%accuracy
	
	
	
	
	
	
	
	
	

