import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC

from processEmail import processEmail
from emailFeatures import emailFeatures
from getVocabList import getVocabList
from collections import OrderedDict

if __name__ == '__main__':
	print "Processing sample email (emailSample1.txt)"
	file=open('emailSample1.txt','r')
	file_contents=file.readlines()
	word_indices=processEmail(''.join(file_contents))

	#print "Word Indices:"
	#print word_indices

	features=emailFeatures(word_indices)
	#print "Length of feature vector:",features.size
	#print "Number of non-zero entries:",np.sum(features)

	print "Loading and optimizing feature parameters using Linear Kernel..."
	data=loadmat('spamTrain.mat')
	X=data['X']
	y=data['y'].flatten()
	clf=SVC(C=0.1,kernel='linear',tol=1e-3)
	model=clf.fit(X,y)
	predictions=model.predict(X)
	accuracy=np.mean(np.double(predictions==y))*100
	print "Accuracy in training data is : %0.2f "%(accuracy)
	
	data=loadmat('spamTest.mat')
	Xtest=data['Xtest']
	ytest=data['ytest'].flatten()
	predictions=model.predict(Xtest)
	accuracy=np.mean(np.double(predictions==ytest))*100
	print "Accuracy in test data is :%0.2f "%(accuracy)

	vocabList=getVocabList()	
	sortlist= sorted(list(enumerate(model.coef_[0])),key=lambda e: e[1], reverse=True)
	orderedDictn=OrderedDict(sortlist) # dictionary that remembers order in which they were added
	idx=orderedDictn.keys()
	weight=orderedDictn.values()

	raw_input("Press any key to print top predictors...")
	print "Top predictors of spam:"
	for i in range(15):
		print "%s (%f)"%(vocabList[idx[i]],weight[i])

	print "\nChecking for some Email..."
	filename='spamSample1.txt'
	file=open(filename)
	file_contents=file.readlines()
	word_indices=processEmail(''.join(file_contents))
	x=emailFeatures(word_indices)
	x=x.reshape(-1,1)
	x=x.reshape(1,1899)
	prediction=model.predict(x)
	print prediction
	print "Processed %s \nSpam Classification: %d "%(filename,prediction)
	print "(1 indicates Spam, 0 indicates NOT Spam"



	
