'''
Copyright 2012 Ashwin Panchapakesan

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

from random import choice as choose, sample
from random import random
import numpy as np
from sklearn.naive_bayes import GaussianNB

from Genetic.individual import Individual

X = []
Y = []



def genPop(N, chromGenfuncs, chromGenParams):
	""" Return a population (list) of N unique individuals.
		Each individual has len(chromgGenFuncs) chromosomes.
		For each individual, chromosome_i is generated by calling chromGenFuncs_i(chromeGenParams_i)
		
		pre:
			N >= 0
			isinstance(chromGenfuncs, list)
			isinstance(chromGenParams, list)
			len(chromGenfuncs) == len(chromGenParams)
		
		post:
			isinstance(__return__, list)
			len(__return__) == N
			forall(__return__, lambda indiv: __return__.count(indiv) == 1)
		
		post[chromGenfuncs, chromGenParams]:
			__old__.chromGenfuncs == chromGenfuncs
			__old__.chromGenParams == chromGenParams
	"""
	
	answer = set()
	chromGens = zip(chromGenfuncs, chromGenParams)
	while len(answer) < N:
		indiv = Individual([])
		for genfunc, genparams in chromGens:
			indiv.append(genfunc())
			indiv.append(0)
		answer.add(indiv)
	return list(answer)

def genCharsChrom():
	""" Return chromosome (list) of length l, each of which is made up of the characters from chars. 
		
		pre:
			isinstance(l, int)
			hasattr(chars, '__getitem__')
			hasattr(chars, '__len__')
			len(chars) > 0
		
		post[l, chars]:
			__old__.l == l
			__old__.chars == chars
			len(__return__) == l
			forall(__return__, lambda a: a in chars)
	"""
	
	l = np.random.random_integers(0,99,20)

	for i in range(0,100):
		if Y[i] == 1:
			#print("!!!!!!")
			l = np.append(l,[i])
			break


	for i in range(0,100):
		if Y[i] == -1:
			#print("######")
			l = np.append(l,[i])
			break

	x = X[l]
	y = Y[l]


	clf =  GaussianNB()

	clf = clf.partial_fit(x,y,[1,-1])

	return clf

'''
	cond = np.random.rand(classes,features)
	priors = np.random.random(classes)

	priors = priors / np.sum(priors)

	sums = np.sum(cond,axis=0)
	cond = cond/sums

	#print(priors)
	#print(cond)

	return cond,priors
'''

def genTour(numCities):
	"""
		pre:
			isinstance(numCities, int)
		
		post[numCities]:
			__old__.numCities == numCities
		post:
			isinstance(__return__, list)
			len(__return__) == numCities
			forall(__return__, lambda c: 0<= c < numCities)
			forall(__return__, lambda c: __return__.count(c)==1)
	"""
	return sample(range(numCities), numCities)
