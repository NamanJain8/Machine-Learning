import numpy as np
import matplotlib.pyplot as plt
import re

with open('ex1data1.txt') as f:
	content =f.readlines()
content =[x.strip() for x in content]

for i in range(20):
	content[i]=re.findall(r'[-+]*\d*\.\d+',content[i])

content = np.asarray(content)
content=np.transpose(content)
def plotData():
	plt.plot(content[0],content[1],'ro')
	plt.ylabel('Profit in$10,000s')
	plt.xlabel('Population of City in 10,000s')
	plt.show()
	return
