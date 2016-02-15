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

from Genetic import settings, selection, visualization as vis
from Genetic import mutation, crossover, fitness, individual, population #@UnusedImport # for contract checking only
from Genetic.individual import Individual #@UnusedImport # for contract checking only
from random import random as rand
import logging as log
import pandas as pd
import numpy as np
import pygame as pg #@UnresolvedImport
import comparison
import matplotlib.pyplot as plt

#import contract
#for mod in [crossover, fitness, individual, mutation, population, selection, vis]:
#	contract.checkmod(mod)

log.basicConfig(format='%(levelname)s|%(message)s', level=log.DEBUG)



def runTSPGA(kwargs):
	"""
		pre:
			isinstance(kwargs, dict)
			'maxGens' in kwargs
			kwargs['maxGens'] > 0
		
		post[kwargs]:
			__old__.kwargs == kwargs
			__return__[0][1] >= kwargs['targetscore'] or __return__[1] == kwargs['maxGens']
			isinstance(__return__[0][0], Individual)
	"""
	
	if 'sanity' not in kwargs:
		raise TypeError("Expected argument 'sanity' not found")
	arguments = kwargs['sanity']
	
	if len(kwargs) < len(arguments):
		raise TypeError("Missing Arguements: %s" %' '.join([a for a in arguments if a not in kwargs]))

	# # # # # # PARAMETERS # # # # # #
	
	testmode = kwargs['testmode']
	
	maxGens = kwargs['maxGens']
	targetscore = kwargs['targetscore']
	genfunc = kwargs['genfunc']
	genparams = kwargs['genparams']

	scorefunc = kwargs['scorefunc']
	scoreparams = kwargs['scoreparams']

	selectfunc = kwargs['selectfunc']
	selectparams = kwargs['selectparams']
	
	numcross = kwargs['numcross']
	crossfunc = kwargs['crossfunc']
	crossprob = kwargs['crossprob']
	crossparams = kwargs['crossparams']

	mutfunc = kwargs['mutfunc']
	mutprob = kwargs['mutprob']
	mutparams = kwargs['mutparams']
	
	SCORES = kwargs['SCORES']
	visualize = kwargs['visualize']
	getWheel = kwargs['getWheel']
	
	if visualize:
		makeScreenParams = kwargs['makeScreenParams']
		drawParams = kwargs['drawParams']
		font = kwargs['font']
		fontParams = kwargs['fontParams']
		labelParams = kwargs['labelParams']

	# # # # # # /PARAMETERS # # # # # #
	
	pop = genfunc(*genparams)
	for p in pop:
		if p not in SCORES:
			SCORES[p] = scorefunc(p, *scoreparams)
	
	best = max(SCORES, key=SCORES.__getitem__)
	best = best, SCORES[best]	# indiv, score
	
	if visualize:
		screen = vis.makeScreen(*makeScreenParams)
		label = font.render("%d / %d" %(best[1], targetscore), *fontParams)
		screen.blit(label, *labelParams)
		pg.display.init()
		vis.draw(best[0], screen, *drawParams)
	
	g = 0
	while g < maxGens:
		if testmode:
			assert g < maxGens
			assert best[1] < targetscore
			
		if getWheel:
			wheel = selection.getRouletteWheel(pop, SCORES)
		
		newpop = []
		for _ in xrange(numcross):
			if getWheel:
				p1 = selectfunc(wheel, *selectparams)
				p2 = selectfunc(wheel, *selectparams)
			else:
				p1, p2 = selectfunc(pop, *selectparams)
			if rand() <= crossprob:
				c1 = crossfunc(p1, p2, *crossparams)
				c2 = crossfunc(p2, p1, *crossparams)
				newpop.extend([c1,c2])
		
		for i,p in enumerate(newpop):
			if rand() <= mutprob:
				newpop[i] = mutfunc(p, *mutparams)
				p = newpop[i]
			SCORES[p] = scorefunc(p, *scoreparams)
		
		pop = sorted(pop+newpop, key=SCORES.__getitem__, reverse=True)[:len(pop)]
		
		fittest = max(pop, key=SCORES.__getitem__)
		fittest = fittest, SCORES[fittest]
		log.info("Generation %03d | highest fitness: %s | fittest indiv: %r" %(g, fittest[1], fittest[0].chromosomes[0]) )
		
		if fittest[1] > best[1]:
			best = fittest
			if visualize:
				screen = vis.makeScreen(*makeScreenParams)
				label = font.render("%d / %d" %(best[1], targetscore), *fontParams)
				screen.blit(label, *labelParams)
				pg.display.init()
				vis.draw(fittest[0], screen, *drawParams)
		
			if best[1] >= targetscore:
				if visualize:
					raw_input("Hit <ENTER> to kill visualization: ")
					vis.killscreen()
				
				return best[0], g
		g += 1
	if visualize:
		raw_input("Hit <ENTER> to kill visualization: ")
		vis.killscreen()
	
	if testmode:
		assert (g == maxGens) or best[1] >= targetscore

	return best, g

def runGA(kwargs, testmode=False):
	"""
		pre:
			isinstance(kwargs, dict)
			'maxGens' in kwargs
			kwargs['maxGens'] > 0
		
		post[kwargs]:
			__old__.kwargs == kwargs
			__return__[0][1] >= kwargs['targetscore'] or __return__[1] == kwargs['maxGens']
			isinstance(__return__[0][0], Individual)
	"""
	
	if 'sanity' not in kwargs:
		raise TypeError("Expected argument 'sanity' not found")
	arguments = kwargs['sanity']
	
	if len(kwargs) < len(arguments):
		raise TypeError("Missing Arguements: %s" %' '.join([a for a in arguments if a not in kwargs]))
	
	# # # # # # PARAMETERS # # # # # #
	
	maxGens = kwargs['maxGens']
	targetscore = kwargs['targetscore']
	genfunc = kwargs['genfunc']
	genparams = kwargs['genparams']

	scorefunc = kwargs['scorefunc']
	scoreparams = kwargs['scoreparams']

	selectfunc = kwargs['selectfunc']
	selectparams = kwargs['selectparams']
	
	numcross = kwargs['numcross']
	crossfunc = kwargs['crossfunc']
	crossprob = kwargs['crossprob']
	crossparams = kwargs['crossparams']

	mutfunc = kwargs['mutfunc']
	mutprob = kwargs['mutprob']
	mutparams = kwargs['mutparams']
	
	SCORES = kwargs['SCORES']
	diff = kwargs['diff']
	getWheel = kwargs['getWheel']

	# # # # # # /PARAMETERS # # # # # #
	

	df = pd.read_csv('Genetic/elecTrainData.csv',header=False)

	dfY = df.iloc[:,9]
	dfX = df.iloc[:,1:9]

	
	num_classes = 2
	num_features = 8

	genparams = (genparams[0],genparams[1],[(8,2)])


	crossover.X = dfX.iloc[0:100].values
	crossover.Y = dfY.iloc[0:100].values


	population.X = dfX.iloc[0:100].values
	population.Y = dfY.iloc[0:100].values


	pop = genfunc(*genparams)

	N = len(pop)

	

	start = 100
	end  = start + diff

	fitness.currX = dfX.iloc[start:min(end,len(dfX))].values
	fitness.currY = dfY.iloc[start:min(end,len(dfX))].values


	for p in pop:
		if p not in SCORES:
			SCORES[p] = scorefunc(p, *scoreparams)
	
	best = max(SCORES, key=SCORES.__getitem__)
	best = best, SCORES[best]	# indiv, score
	
	g = 1

	acc = []

	while  g < maxGens and end < len(dfX):
		if testmode:
			assert g < maxGens
			assert best[1] < targetscore

		if getWheel:
			wheel = selection.getRouletteWheel(pop, SCORES)
		
		newpop = []
		for _ in xrange(numcross):
			
			if getWheel:
				p1 = selectfunc(wheel, *selectparams)
				p2 = selectfunc(wheel, *selectparams)
			else:
				p1, p2 = selectfunc(pop, *selectparams)
			
			if rand() <= crossprob:
				l = crossfunc(p1,p2,g,SCORES[p1],SCORES[p2])
				
				for a in l:
					SCORES[a] = scorefunc(a, *scoreparams)	

				newpop.extend(l)			
				
			else:
				newpop.extend([p1,p2])			

		
		for i,p in enumerate(newpop):
			if rand() <= mutprob:
				newpop[i] = mutfunc(p,SCORES[p],crossover.Y)
				p = newpop[i]
			SCORES[p] = scorefunc(p, *scoreparams)
		
		pop = newpop

		sorted(pop,key = fitness.scoreAccuracy)

		pop = pop[0:N]

		#print(len(pop))

		crossover.X = fitness.currX 
		crossover.Y = fitness.currY


		fitness.currX = dfX.iloc[start:min(end,len(dfX))].values
		fitness.currY = dfY.iloc[start:min(end,len(dfX))].values

		
		fittest = max(pop, key=SCORES.__getitem__)
		fittest = fittest, SCORES[fittest]
		#log.info("Generation %03d | highest fitness: %s | fittest indiv: %r" %(g, fittest[1], ''.join(fittest[0].chromosomes[0])) )
		a = fittest[0].chromosomes[0].score(fitness.currX,fitness.currY)
		#print("Generation %03d | highest fitness: %s | Born: %d | prediction: %s" %(g, fittest[1],fittest[0].chromosomes[1],a))
		
		acc.append(a) 
		
		if fittest[1] > best[1]:
			best = fittest
			if best[1] >= targetscore:
				return best, g

		g += 1


		fitness.evolution(pop)

		start += diff
		end += diff

	
	if testmode:
		assert (g == maxGens) or best[1] >= targetscore
	

	return acc


def disp(dic):


	for key,val in dic:
		print(key + " " + sum(val)/len(val) + "\n")



if __name__ == "__main__":
	print 'starting'
#	contract.checkmod(__name__)
	

	dic = {}
	lis = []
	nam = []
	for setting,name in settings.listOfSettings():

		answer = runGA(setting)
		print(name + "," + str(sum(answer)/len(answer)) + "\n")
		

		l = range(1,len(answer)+1)	
		plo, = plt.plot(l,answer,label=name)

		nam.append(name)
		lis.append(plo)


	
	m = 500


	answer = comparison.comp(50,m)

	name = "Comparison50"
	l = range(1,len(answer)+1)	
	plo, = plt.plot(l,answer,label=name)

	nam.append(name)
	lis.append(plo)

	print(name + "," + str(sum(answer)/len(answer)) + "\n")


	'''
	answer = comparison.comp(100,m)


	name = "Comparison100"
	l = range(1,len(answer)+1)	
	plo, = plt.plot(l,answer,label=name)

	nam.append(name)
	lis.append(plo)

	print(name + "," + str(sum(answer)/len(answer)) + "\n")

	answer = comparison.comp(200,m)

	name = "Comparison200"
	l = range(1,len(answer)+1)	
	plo, = plt.plot(l,answer,label=name)

	nam.append(name)
	lis.append(plo)

	print(name + "," + str(sum(answer)/len(answer)) + "\n")
	
	#print(lis)
	'''

	plt.legend(lis,nam)
	plt.show()
	print 'done'
