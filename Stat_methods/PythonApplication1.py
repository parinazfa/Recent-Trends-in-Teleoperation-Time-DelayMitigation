import sys
import numpy as np
import matplotlib.pyplot as plt
import CosineSimilarity
import StatisticalPredictionMethods
import pandas as pd
from random import random
from mpl_toolkits.mplot3d import Axes3D
# A. Pandya, Statisitical processing

###########
#Input Data
###########
#Uncomment one of the following data sets to use

# contrived very simple linear dataset
#data= pd.DataFrame(np.zeros(shape =(100, 3)))
#data.iloc[:,0] =  [r  for r in range(0, 100)]
#data.iloc[:,1] =  [r  for r in range(0, 100)]
#data.iloc[:,2] =  [r  for r in range(0, 100)]

##Create an Euler set of 3d Point (Spiral) Z = cos(t) + isin(t), without noise
#data= pd.DataFrame(np.zeros(shape =(100, 3)))
#x = 0
#for i in range(100):
#    data.iloc[i] = [x, np.cos(x) , np.sin(x)]
#    x = x +.1


##Create an Euler set of 3d Point (Spiral) Z = cos(t) + isin(t), with noise
#data= pd.DataFrame(np.zeros(shape =(100, 3)))
#x = 0
#for i in range(100):
#    data.iloc[i] = [x, np.cos(x)+ (np.random.random_sample()*.1), np.sin(x)+(np.random.random_sample()* .1)]
#    x = x +.1

#Read a small amout of rows of data from a file of values
data = pd.read_csv("data.csv", header=None, nrows = 500)  #note, this is not the whole file

################### 
# Program set up
###################

#how many values to use (npreddata) and how many to forcast (forcastdata)
npredata = 20  #how much predata must be used 
forcastdata = 10 #how many steps of forcast is needed
#1: self.AutoRegression,
#2: self.MovingAverage,
#3: self.AutoRegressinveMovingAverage,
#4: self.AutoRegressiveIntegeratedMovingAverage,
#untested:
#5: self.SeasonalAutoRegressiveIntegratedMovingAverage,
#6: self.SeasonaAutoIntegratedMovingAverageExogenousRegressors,
#7: self.VectorAutoRegressive
method = 2  # use methods 1-8 that are implemented in StatisicalPredictionMethods

#Create a variable of StatisticalPredictionMethods
p = StatisticalPredictionMethods.StatisticalPredictionMethods(data, forcastdata )

#calculate the error for each prediction segment
sumerror= 0
#for i in range(len(data)-npredata):
for i in range (int(len(data)/npredata)-1 ):  #Break data up into pieces of npredata
    
    #get the predata segment from actual data (ground truth)
    predata = data.iloc[i: i+ npredata, :]

    #Assign that data for prediction, and assign also how much to forcast
    p.AssignData(predata, forcastdata)
    
    #predict using one of many algorithms (1 = AutoRegression etc.)
    predicted = p.Predict(method)  #choose a particular method

    #This maintains a list of all the predictions for later display
    if (i ==0):
        segmented_predictions = predicted.copy()
    else:
        #append all the segments
        segmented_predictions = segmented_predictions.append(predicted , ignore_index=True)

    #extract the actual data (what it should have predicted) 
    actual = data.iloc[(i+npredata):(i+npredata+forcastdata), :]

    #compute the error for each segment and add it.
    sumerror = sumerror + p.ComputeError(actual, predicted)

#print the error average
print ('mean sum of all error = ', np.mean(sumerror))

#plot the original and segmented data 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#original graph ploted in red
ax.scatter(data.iloc[:,0],data.iloc[:,1], data.iloc[:,2], c ='r', marker = '.')

#predicted vaues plotted in green
ax.scatter(segmented_predictions.iloc[:,0], segmented_predictions.iloc[:,1], segmented_predictions.iloc[:,2], c = 'g', marker = 'x')

plt.show()
