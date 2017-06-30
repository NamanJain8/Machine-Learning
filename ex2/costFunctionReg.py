import numpy as np
import costFunction as cf

def costFunctionReg(theta,X,y,_lambda):
	m=X.shape[0]
	cost,grad=cf.costFunction(theta,X,y)
	reg_cost=_lambda*(1/2.0*m)*sum(theta**2)
	reg_grad=_lambda*(1/m)*sum(theta[1:])
	return cost+reg_cost,grad+reg_cost

