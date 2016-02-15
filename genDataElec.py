import numpy as numpy
import pandas as pd
from numpy import random as r

df =  pd.read_csv("elecNormNew.csv",header=False)

f = lambda x : 1 if x == "UP" else -1

df['class'] = df['class'].map(f)

X = df.values

df.to_csv("Genetic/elecTrainData.csv")
