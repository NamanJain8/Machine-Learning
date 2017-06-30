import matplotlib.pyplot as plt
import numpy as np
import plotData as pd

def plotDecisionBound(theta,X,y):
	pd.plotData(X[:,1:],y,show=False)
	plot_x=np.array([X[:,1].min()-2,X[:,1].max()+2])
	plot_y=np.array(-theta[0]-theta[1]*plot_x)/theta[2]
	plt.plot(plot_x,plot_y)
	plt.show()


