import warmUpExercise as wue
import plotData as pd
import computeCost as cc
import featureNormalize as fn
import gradientDescent as gd
import normalEq as ne
import numpy as np
import matplotlib.pyplot as plt

print "Running warmUpExercise"
print "5*5 identity matrix"
wue.warmUpExercise()
raw_input("Press Enter to continue.....")

print "Plotting Data"
pd.plotData()
raw_input("Press Enter to continue.....")

a=pd.content[0]
X=np.ones((20,2))
X[:,1]=a
b=pd.content[1]
y=np.ones((20,1))
y[:,0]=b

theta=np.zeros((2,1))
iterations=100
alpha= 0.01

J=cc.computeCost(X,y,theta)
print "Computed Cost: ",J
raw_input("Compute theta?")

theta=gd.gradientDescent(X,y,theta,alpha,iterations)
print "Theta found by gradient descent: ",theta
raw_input("Press Enter to plot linear graph.....")

val=np.arange(4,14,0.1)
pred=[theta[0]+theta[1]*x for x in val]
plt.plot(a,b,'ro')
plt.plot(val,pred)
plt.ylabel("Profit")
plt.xlabel("Population")
plt.show()

