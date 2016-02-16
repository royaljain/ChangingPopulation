from random import random as rand
import logging as log
import pandas as pd
import numpy as np
import pygame as pg

from random import choice as choose, sample
from random import random
import numpy as np
from sklearn.naive_bayes import GaussianNB


def comp(diff,maxGen):

	df = pd.read_csv('Genetic/Data/simSEAData.csv',header=False)

	dfY = df.iloc[:,3]
	dfX = df.iloc[:,0:3]

	#print(dfY)
	#print(dfX)


	X = dfX.iloc[0:100].values
	Y = dfY.iloc[0:100].values

	l = np.random.random_integers(0,50,20)

	for i in range(0,50):
		if Y[i] == 1:
			#print("!!!!!!")
			l = np.append(l,[i])
			break


	for i in range(0,50):
		if Y[i] == -1:
			#print("######")
			l = np.append(l,[i])
			break

	x = X[l]
	y = Y[l]


	clf =  GaussianNB()

	clf = clf.partial_fit(x,y,[-1,1]	)

	start = 100
	end = start + diff


	currX = dfX.iloc[start:min(end,len(dfX))].values
	currY = dfY.iloc[start:min(end,len(dfX))].values

	i = 0 
	acc = []

	while i < maxGen and end < len(dfX):

		#print(str(i))
		#print("################################################################")
		a = clf.score(currX,currY)
		#print(a)
		acc.append(a)
		#print(clf.theta_)
		clf = clf.partial_fit(currX,currY)
		start += diff
		end += diff

		currX = dfX.iloc[start:min(end,len(dfX))].values
		currY = dfY.iloc[start:min(end,len(dfX))].values
		i = i+1

	return acc


