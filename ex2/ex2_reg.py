import numpy as np
import plotData as pd
import mapFeature as mf
import costFunctionReg as cfr
import matplotlib.pyplot as plt
from scipy.optimize import minimize

if __name__=='__main__':
	data2=np.loadtxt('ex2data2.txt',delimiter=',')
	X_original=X=data2[:,:2]
	y=data2[:,2]
	pd.plotData(X,y,True,reg=True)

	raw_input("Press Enter to create new features")	
	X=mf.mapFeature(X[:,0],X[:,1])
	print "New features created"

	m,n=X.shape
	_lambda=1.0
	initial_theta=np.zeros(n)
	cost,grad=cfr.costFunctionReg(initial_theta,X,y,_lambda)
	print "Cost calculated for initial theta: %f"%cost
	print "Grad calculated for initial theta:\n%s "%grad
	raw_input("Press Enter to optimise theta.....")

	objective=lambda t : cfr.costFunctionReg(t,X,y,_lambda)[0]

	result=minimize(objective,initial_theta,method='CG',options={'maxiter':100000,'disp':False,})
	theta=result.x
	print theta
	raw_input("Press Enter to plot decision Boundary......")
	
	pd.plotData(X_original,y,show=False,reg=True)
	u=np.linspace(-1,1.5,100)
	v=np.linspace(-1,1.5,100)
	z=np.zeros((u.size,v.size))
	for i in range(u.size):
		for j in range(v.size):
			z[i,j]=(mf.mapFeature(u[i],v[j])).dot(theta)
	print z
	plt.contour(u,v,z.T,[-0.02,-0.01,-0.007,-0.005,-0.0025])
	plt.show()

