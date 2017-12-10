import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC

def plotData(X,y):
	pos=(y==1).nonzero()
	neg=(y==0).nonzero()
	plt.plot(X[pos,0],X[pos,1],'b+')
	plt.plot(X[neg,0],X[neg,1],'yo')
	plt.show()

def plot_svc(svc,X,y,h=0.02,pad=0.25):
	x_min,x_max=X[:,0].min()-pad,X[:,0].max()+pad
	y_min,y_max=X[:,1].min()-pad,X[:,1].max()+pad
	xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
	Z=svc.predict(np.c_[xx.ravel(),yy.ravel()])
	Z=Z.reshape(xx.shape)
	plt.contour(xx,yy,Z,alpha=0.2)
	plotData(X,y)
	#plots position of support vectors
	sv = svc.support_vectors_
	plt.scatter(sv[:,0], sv[:,1], c='k', marker='o')
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()
	#print('Number of support vectors: ', svc.support_.size) #to print no. of support vectors used

def gaussianKernel(x1,x2,sigma=2.0):
	norm=np.sum((x1-x2)**2)
	return (np.exp(-norm/(2*sigma**2)))

def dataset3Params(X,y,Xval,yval):
	Cval,gval=0.01,1000
	Cbest,gval=1,1
	accuracy_best=0
	for i  in range(10):
		Cval=Cval*3
		for j in range(10):
			gval=gval*3
			clf=SVC(C=Cval,kernel='rbf',gamma=gval)
			clf.fit(X,y.flatten())
			predictions=clf.predict(Xval)
			accuracy=np.mean(1.0*(predictions==yval))*100
			if accuracy>accuracy_best:
				accuracy_best=accuracy
				gbest=gval
				Cbest=Cval
		gval=1000

	return Cbest,gbest,accuracy_best

if __name__ == '__main__':
	print "Loading and visualising data1..."
	data1=loadmat('ex6data1.mat')
	X=data1['X']			#shape(51,2)
	y=data1['y']			#shape(51,1)
	#plotData(X,y)
	#raw_input("Press any key to implement Linear kernel...")
	clf=SVC(1.0,kernel='linear')
	clf.fit(X,y.flatten())
	#plot_svc(clf,X,y)

	print "Implementing Gaussian..."
	x1=np.array([1,2,1])
	x2=np.array([0,4,-1])
	sigma=2.0
	print gaussianKernel(x1,x2,sigma)

	print "Loading and visualising data2..."
	data2=loadmat('ex6data2.mat')
	X=data2['X']
	y=data2['y']
	#plotData(X,y)
	#raw_input("Press any key to implement Gaussian kernel...")
	clf=SVC(50,kernel='rbf',gamma=6)
	clf.fit(X,y.flatten())
	#plot_svc(clf,X,y)

	print "Loading and visualising data3..."
	data3=loadmat('ex6data3.mat')
	X=data3['X']
	y=data3['y']
	Xval=data3['Xval']
	yval=data3['yval']
	#plotData(X,y)
	print "Calculating parameters..."
	C_,gamma_,accuracy=dataset3Params(X,y,Xval,yval)
	#raw_input("Press any key to plot data")
	clf=SVC(C=C_,kernel='rbf',gamma=gamma_)
	clf.fit(X,y.flatten())
	plot_svc(clf,X,y)