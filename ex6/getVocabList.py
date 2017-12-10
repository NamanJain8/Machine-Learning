def getVocabList():
	f=open('vocab.txt')
	vocabList=[]
	for line in f:
		idx,w=line.split()
		vocabList.append(w)
	f.close()
	return vocabList