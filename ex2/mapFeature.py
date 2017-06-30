import numpy as np

def mapFeature(x1,x2,degree=6):
	m=x1.shape[0] if x1.shape else 1
	cols=[np.ones((m))]
	
	for i in range(1,degree+1):
		for j in range(i+1):
			cols.append((x1**(i-j))*(x2**j))
	return np.vstack(cols).T	
#concatenate could also be used in place of vsatck

