from scipy.io import loadmat
import numpy as np 
import matplotlib.pyplot as plt
import PIL.Image as img
#import imageio as img

def displayData(X,idx,K):
	plt.xlabel("Parameter 1")
	plt.ylabel("Parameter 2")
	lab1=(idx==1).nonzero()[0]
	lab2=(idx==2).nonzero()[0]
	lab3=(idx==3).nonzero()[0]
	plt.plot(X[lab1,0],X[lab1,1],'ro')
	plt.plot(X[lab2,0],X[lab2,1],'go')
	plt.plot(X[lab3,0],X[lab3,1],'bo')
	plt.show()
	return

def findClosestCentorid(X,centroids):
	m=X.shape[0]
	K=centroids.shape[0]
	idx=np.ones((m,1))
	temp=np.zeros((m,K))
	for i in range(K):
		temp[:,i]=np.sum(((X-centroids[i])**2),axis=1)
	idx=1+np.argmin(temp,axis=1)
	return idx

def computeCentroids(X,idx,K):
	m,n=X.shape
	centroids=np.zeros((K,n))
	count=np.zeros((K,1))
	ssum=np.zeros((K,n))
	for i in range(m):
		count[idx[i]-1]+=1
		ssum[idx[i]-1]+=X[i]
	for i in range(K):
		centroids[i]=ssum[i]/count[i]
	return centroids

def runKMeans(X,centroids,K,max_iter,show=False):
	idx=np.ones((X.shape[0],1))
	for i in range(max_iter):
		idx=findClosestCentorid(X,centroids)
		centroids=computeCentroids(X,idx,K)
		if(show==True):
			displayData(X,idx,K)
	return centroids,idx

def kMeansInitCentroids(X,K):
	centroids=np.random.permutation(X)[:K,:]	
	return centroids


if __name__=="__main__":
	print "Loading Data1"
	data=loadmat('ex7data2.mat')
	X=data['X']
	X=np.asarray(X)
	K=3
	initial_centroids=np.asarray([[3,3],[6,2],[8,5]])
	
	idx=findClosestCentorid(X,initial_centroids)
	print "Centroids for first 3 examples are:", idx[:3]
	raw_input("Press Enter to continue....")

	centroids=computeCentroids(X,idx,K)
	print centroids
	centroids,idx_KMeans=runKMeans(X,initial_centroids,K,10)	#run K-Means Algo

	image=img.open('bird_small.png')
	original_img=image
	image=np.asarray(image)/255.0
	image=image.reshape(image.shape[0]*image.shape[1],3)
	K=16
	max_iters=10
	initial_centroids=kMeansInitCentroids(image,K)
	centroids,idx_KMeans=runKMeans(image,initial_centroids,K,max_iters)

	raw_input("Applying K-Means to compress image\
		Press any Key to see....")
	X_recovered=centroids[idx_KMeans-1,:]
	X_recovered=X_recovered.reshape(original_img.size[0],original_img.size[1],3)
	rec_img=img.fromarray(X_recovered,'RGB')
	
	plt.subplot(1,2,1)
	plt.imshow(original_img)
	plt.title("Original")

	plt.subplot(1,2,2)
	plt.imshow(X_recovered)
	plt.title("Compressed")

	plt.show()	

