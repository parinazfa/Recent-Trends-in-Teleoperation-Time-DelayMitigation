import sys
import numpy as numpy
from numpy import dot
from numpy.linalg import norm
import queue

class CosineSimilarity:
    """This class will determine the similarity to previous streams"""

    
    def __init__(self,  x,y,z):
        self.size = len(x)
        self.x = x
        self.y = y
        self.z = z    
        
    def CompareToken(self, token):
        #Compare the given token (which has n 3D data points) for instance
        # token would be [ x1 x2 x3 x4 y1 y2 y3 y4 z1 z2 z3 z4] where all these 
        # are numbers of the xyz plot.  Here tokensize is 4.

        tokensize =len(token)/3 # assuming 3 axes.

        #space for the cosine array
        cos_sim = numpy.zeros(self.size)

        #Compare each set of values sequentially 
        for i in range (self.size):
        
            if (i <= self.size - tokensize):
                tsize = int(i + tokensize )
                b = numpy.concatenate ((self.x[i:tsize], self.y[i:tsize], self.z[i:tsize]), axis =0)
                cos_sim[i] = dot(token, b)/(norm(token)*norm(b))
                 
        #return the maximum index and the 
        index = numpy.argmax(cos_sim)
        maxvalue = cos_sim[index]
        return index, maxvalue
        
    def PredictNextNvalues (self, token, n):
        #Given a token for comparison (3 axis)
        #the function returns the next predicted values from the
        #history
        index, confidence = self.CompareToken(token)
        index = int(index + len(token)/3)
        prediction = numpy.zeros([n, 3])
        j = 0
        #copy the next n predictions after index and the token values
        for i in range (index,index+n,1):
            if (i < self.size):
                prediction[j,0] =self.x[i]
                prediction[j,1] =self.y[i]
                prediction[j,2] =self.z[i]
                j= j+1
            else:
                break

        #return the prediction
        return prediction, confidence        
        

#Create a variable of type CosineSimilatiry
#c = CosineSimilarity.CosineSimilarity(x,y,z)
        
      


