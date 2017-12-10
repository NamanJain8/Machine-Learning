import numpy as np
def emailFeatures(word_indices):
	n=1899
	x=np.zeros(n)
	for index in word_indices:
		x[index]=1
	return x