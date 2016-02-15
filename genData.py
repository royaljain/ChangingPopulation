import numpy as numpy
from numpy import random as r


c = 2
examples = 20000

features = 10

start  = []

for i in range(0,features):

	if r.random() < 0.5 :
		start.append(0.2)
	else:
		start.append(0.8)



end = [ 1 - s for s in start]


limits = zip(start,end)

print(start)
print(end)

curr = start



start1  = []

for i in range(0,features):

	if r.random() < 0.5 :
		start1.append(0.2)
	else:
		start1.append(0.8)



end1 = [ 1 - s for s in start1]


limits1 = zip(start1,end1)

print(start1)
print(end1)

curr1 = start1


mat = numpy.zeros((examples,features+1))

for i in range(0,examples):

	if r.random() >= 0.5:
		mat[i][0] = 1

		for j in range(1,features+1):

			if r.random() < curr[j-1]:
				mat[i][j] = 1.0
			else:
				mat[i][j] = 0.0


	else:
		mat[i][0] = -1

		for j in range(1,features+1):

			if r.random() < curr1[j-1]:
				mat[i][j] = 1.0
			else:
				mat[i][j] = 0.0


	curr = [ ((examples-i)*a + i*b)/(2*examples) for a,b in limits]
	curr1 = [ ((examples-i)*a + i*b)/(2*examples) for a,b in limits1]


numpy.savetxt("Genetic/simTrainData.csv",mat,fmt="%0.1f",delimiter=",")