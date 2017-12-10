import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib import cm
import math


def estimateGaussian(X):
	mu=np.mean(X,axis=0)
	sigma2=np.mean((X-mu)**2,axis=0)
	return mu,sigma2

def plotData(x,mu,sigma2,p,show=False,anomaly=False):
	fig=plt.figure()
	ax=fig.add_subplot(111)
	u=np.linspace(0,25,1000)
	x,y=np.meshgrid(u,u)
	a1=(1.0/np.sqrt(2*math.pi*sigma2[0])) * np.exp( -(x-mu[0])**2/(2*sigma2[0]) )
	a1=a1[0,:]
	a2=(1.0/np.sqrt(2*math.pi*sigma2[1])) * np.exp( -(y-mu[1])**2/(2*sigma2[1]) )
	a2=a2[:,0]
	z=np.outer(a1,a2)
	ax.contour(x,y,z,colors='r')

	plt.xlabel("Latency")
	plt.ylabel("Throughput")
	plt.plot(X[:,0],X[:,1],'bx')
	if anomaly==True:
		anom= (p<epsilon)==1
		plt.plot(X[anom,0],X[anom,1],'ro')

	if show==True:
		plt.show()
	return


def selectThreshold(p,y):
	best_epsilon=1
	best_f1_score=0
	stepsize=(max(p)-min(p))/1000
	
	for i in range(1000):
		epsilon=stepsize*i
		y_cal = p<epsilon
		precision=0
		recall=0
		f1_score=0

		tp= np.sum(np.multiply(y_cal,y))
		tn= np.sum(np.multiply((y_cal==0),(y==0)))
		fp= np.sum(np.multiply(y_cal,(y==0)))
		fn= np.sum(np.multiply((y_cal==0),y))

		if (tp+fp)!=0 and (tp+fn)!=0:
			precision=(1.0*tp)/(tp+fp)
			recall=(1.0*tp)/(tp+fn)
			f1_score=2*recall*precision/(recall+precision)
		
		if best_f1_score<f1_score:
			best_f1_score=f1_score
			best_epsilon=epsilon

	return best_epsilon,best_f1_score

if __name__ == '__main__':
	print "Loading and visualising data..."
	data1=loadmat('ex8data1.mat')
	X=data1['X']
	plt.xlabel("Latency")
	plt.ylabel("Throughput")
	plt.plot(X[:,0],X[:,1],'bx')
	#plt.show()

	mu,sigma2=estimateGaussian(X)

	print "Plotting Contours using Gaussian estimation...."
	plotData(X,mu,sigma2,X)

	print "Estimating parameter epsilon...."
	Xval=data1['Xval']
	yval=data1['yval'].flatten()
	a1=(1.0/np.sqrt(2*math.pi*sigma2[0])) * np.exp( -(Xval[:,0]-mu[0])**2/(2*sigma2[0]) )
	a2=(1.0/np.sqrt(2*math.pi*sigma2[1])) * np.exp( -(Xval[:,1]-mu[1])**2/(2*sigma2[1]) )
	p=a1*a2
	epsilon,f1_score=selectThreshold(p,yval)
	
	print "Anomalous points are encircled..."
	plotData(X,mu,sigma2,p,True,True)
