import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
def featureNormalise(X):
	m=X.shape[0]
	mean=np.mean(X,axis=0)
	sigma_dev=np.std(X,axis=0)
	X=(X-mean)/(sigma_dev)
	return X

def pca(X):
	m,n=X.shape
	covariance_matrix=sigma=(1.0/m)*np.matmul(X.T,X)
	U,S,V=np.linalg.svd(sigma)
	return U,S

def projectData(X,U,K):
	U_reduce=U[:,:K]
	Z=np.matmul(X,U_reduce)
	return Z

def recoverData(Z,U,K):
	m=Z.shape[0]
	n=U.shape[0]
	U_reduce=U[:,:K]
	X_recovered=np.zeros((m,n))
	X_recovered=np.matmul(Z,U_reduce.T)
	return X_recovered

def displayData(sel):
	m,n=sel.shape
	display_rows=int(np.around(np.sqrt(m)))
	display_cols=int(np.around(m/display_rows))
	example_rows=int(np.around(np.sqrt(n)))
	example_cols=int(np.around(n/example_rows))
	#print display_rows,display_cols
	display_mat=np.zeros((display_rows*example_rows,display_cols*example_cols))
	for i in range(display_rows):
		for j in range(display_cols):
			idx=i*display_cols+j
			image=sel[idx]
			image=image.reshape(example_rows,example_cols)
			display_mat[i*example_rows:(i+1)*example_rows,j*example_cols:(j+1)*example_cols]=image
	#plt.imshow(display_mat.T,cmap='gray')
	#plt.show()
	return display_mat


if __name__=="__main__":
	data=loadmat('ex7data1.mat')
	X=data['X']
	X=np.asarray(X)
	plt.plot(X[:,0],X[:,1],'ro')
	plt.show()

	X=featureNormalise(X)
	U,S=pca(X)
	plt.plot(X[:,0],X[:,1],'ro')
	x=np.arange(-0.3,0.3,0.1)
	y=(U[1][0]/U[0][0])*x
	plt.plot(x,y)
	plt.show()
	print "First Eigenvector is: ",U[:,0]
	Z=projectData(X,U,1)
	X_recovered=recoverData(Z,U,1)
	
	print "Loading and optimising parameters using PCA..."
	face=loadmat('ex7faces.mat')
	X=face['X']
	sel1=X[:100,:]
	mat1=displayData(sel1)

	X=featureNormalise(X)
	U,S=pca(X)
	Z=projectData(X,U,36)
	X_recovered=recoverData(Z,U,36)
	sel2=X_recovered[:100,:]
	mat2=displayData(sel2)

	raw_input("Press any key to visualise...")
	plt.subplot(1,2,1)
	plt.title('Original')
	plt.imshow(mat1.T,cmap='gray')
	plt.subplot(1,2,2)
	plt.title('Compressed')
	plt.imshow(mat2.T,cmap='gray')
	plt.show()


