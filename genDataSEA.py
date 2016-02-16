import numpy as numpy
from numpy import random 


examples = 50001

features = 3


mat = numpy.zeros((examples,features+1))

for i in range(0,examples):

	x1 = random.random()*10.0;
	x2 = random.random()*10.0;
	x3 = random.random()*10.0; 

	mat[i][0] = x1
	mat[i][1] = x2
	mat[i][2] = x3

	if i < 12500:
		b = 8

	elif i < 25000:
		b = 9

	elif i < 37500:
		b = 7

	else:
		b = 9.5



	if x1 + x2 <= b: 
		mat[i][3] = 1
	else:
		mat[i][3] = -1



numpy.savetxt("Genetic/Data/simSEAData.csv",mat,fmt="%0.2f",delimiter=",")