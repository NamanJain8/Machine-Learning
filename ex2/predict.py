import numpy as np
import sigmoid as s
def predict(theta,X):
	return s.sigmoid(X.dot(theta))>=0.5
