import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import fmin_cg

def checkCostFunction(params,y,r,num_users,num_movies,num_features,lambda_):
	eps=0.001
	nparams=params.shape[0]
	eps_params=np.zeros(nparams)
	max_diff=0
	# cofiCostFunction Calculated
	grad_cofi=cofiGradFunction(params,y,r,num_users,num_movies,num_features,0)
	# Numerically Calculated
	for i in range(10):
		idx=np.random.randint(0,nparams)
		eps_params[idx]=eps
		loss1=cofiCostFunction(params- eps_params,y,r,num_users,num_movies,num_features,0)
		loss2=cofiCostFunction(params+ eps_params,y,r,num_users,num_movies,num_features,0)
		grad_num=(loss2-loss1)/(2*eps)
		eps_params[idx]=0
		diff=abs(grad_num - grad_cofi[idx])
		if max_diff<diff:
			max_diff=diff
	if max_diff < 1e-10:
		print "cofiCostFunction working fine || Error value:%0.15f"%(max_diff)
	else:
		print "cofiCostFunction not working fine || Error value %0.15f"%(max_diff)
	

def cofiCostFunction(params,y,r,num_users,num_movies,num_features,lambda_):
	X=params[:num_features*num_movies].reshape(num_movies,num_features)
	Theta=params[num_features*num_movies:].reshape(num_users,num_features)
	
	J=0
	h=X.dot(Theta.T)
	h=np.multiply(h,r)

	J=0.5*np.sum(np.square(h-y))
	J+=0.5*lambda_*(np.sum(np.square(Theta))+np.sum(np.square(X)))
	
	return J

def cofiGradFunction(params,y,r,num_users,num_movies,num_features,lambda_):
	X=params[:num_features*num_movies].reshape(num_movies,num_features)
	Theta=params[num_features*num_movies:].reshape(num_users,num_features)

	X_grad=np.zeros((num_movies,num_features))
	Theta_grad=np.zeros((num_users,num_features))
	h=X.dot(Theta.T)
	h=np.multiply(h,r)

	X_grad=(h-y).dot(Theta)
	Theta_grad=(h-y).T.dot(X)

	X_grad+=lambda_*X
	Theta_grad+=lambda_*Theta
	grad=np.concatenate((X_grad.flatten(),Theta_grad.flatten()))
	return grad



def normalizeRatings(y,r):
	mean=np.sum(y,axis=1)/np.sum(r,axis=1)
	mean=mean.reshape((y.shape[0],1))
	y=y-mean
	return y,mean

if __name__ == '__main__':
	print "Movie rating dataset..."
	data=loadmat('ex8_movies.mat')
	y=data['Y']
	r=data['R']
	tval= (r[0,:]==1).nonzero()
	print "Average rating for movie 1: Toy Story is  ",np.mean(y[0,tval])

	movie_params=loadmat('ex8_movieParams.mat')
	X=movie_params['X']										#X_shape=1682+10 #theta_shape=943+10
	Theta=movie_params['Theta']

	num_movies=5
	num_users=4
	num_features=3
	params=np.concatenate((X[:5,:3].flatten(),Theta[:4,:3].flatten()))
	J=cofiCostFunction(params,y[:5,:4],r[:5,:4],num_users,num_movies,num_features,0)
	print J

	plt.figure()
	plt.imshow(y, aspect='equal', origin='upper', extent=(0, y.shape[1], 0, y.shape[0]/2.0))
	plt.ylabel('Movies')
	plt.xlabel('Users')
	plt.colorbar()
	plt.show()
	checkCostFunction(params,y[:5,:4],r[:5,:4],num_users,num_movies,num_features,0)


	movies=[]
	file=open('movie_ids.txt')
	for line in file:
		movies.append(' '.join(line.strip('\n').split(' ')[1:]))
	my_rating=np.zeros((X.shape[0]))
	idxs=[0,34,64,123,456]
	my_rating[idxs]=3,4,5,1,2
	my_rating=my_rating.reshape((1682,1))
	num_movies,num_users=y.shape
	r=np.hstack((r,my_rating>0))
	y=np.hstack((y,my_rating))
	num_features=10
	y_norm,mean=normalizeRatings(y,r)
	num_users+=1
	params=np.random.rand(num_features*(num_users+num_movies))
	lambda_=10
	result=fmin_cg(cofiCostFunction,x0=params,args=(y,r,num_users,num_movies,num_features,lambda_),fprime=cofiGradFunction,maxiter=100000,disp=True,full_output=True)
	params=result[0]
	X_cal=params[:num_movies*num_features].reshape(num_movies,num_features)
	Theta_cal=params[num_features*num_movies:].reshape(num_users,num_features)
	prediction=X_cal.dot(Theta_cal.T)
	my_predictions=prediction[:,-1]+mean.flatten()
	pred_idxs_sorted = np.argsort(my_predictions)
	pred_idxs_sorted[:] = pred_idxs_sorted[::-1]	#reverse the order

	print "Top recommendations for you:"
	for i in xrange(10):
	    print 'Predicting rating %0.1f for movie %s.' %(my_predictions[pred_idxs_sorted[i]],movies[pred_idxs_sorted[i]])
	    
	print "\nOriginal ratings provided:"
	for i in xrange(len(my_ratings)):
	    if my_ratings[i] > 0:
	        print 'Rated %d for movie %s.' % (my_ratings[i],movies[i])


	