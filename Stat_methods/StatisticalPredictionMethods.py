import pandas as pd
import numpy as np

#Statistical methods that are implemented 
from statsmodels.tsa.ar_model import AutoReg            
from statsmodels.tsa.arima_model import ARMA            
from statsmodels.tsa.arima_model import ARIMA           
from statsmodels.tsa.statespace.sarimax import SARIMAX  
from statsmodels.tsa.vector_ar.var_model import VAR     
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class StatisticalPredictionMethods:
    ##############################
    #Constructor function
    ##############################
    def __init__(self, data, nforcast):
        #inputs the data, and forcast amount.
        self.data = data        #assign the data
        self.start = len(data)  #start of forcast
        self.end = len(data) +nforcast #stop of forcast

        self.lg = 5 # amount of lag for AR

        #These are all the functions that are defined
        #They are indexed by number to make it easy to call
        self.functions ={
            1: self.AutoRegression,
            2: self.MovingAverage,
            3: self.AutoRegressinveMovingAverage,
            4: self.AutoRegressiveIntegeratedMovingAverage,
            5: self.SeasonalAutoRegressiveIntegratedMovingAverage,
            6: self.SeasonaAutoIntegratedMovingAverageExogenousRegressors,
            7: self.VectorAutoRegressive,
            8: self.VectorAutoRegressiveMovingAverage,
            9: self.HoltWintersExponentialSmoothing

        }
        
    def Predict (self, method_type):
        #find which function to get from the list based on index
        prediction_function = self.functions.get (method_type) 
        
        exodata = [0] #we have to figure the exodata out later.
        datahat = prediction_function (self.data, exodata)

        return datahat

    #####################
    #Assign Data to class
    ###################### 
    def AssignData(self, data, forcastlength):
        self.data = data
        self.start = len(data)
        self.end = self.start + forcastlength

    """This class will predict the next n values based on a particular statistical model"""
   #####################
   #  Error Calculation 
   #####################
    def ComputeError (self, original, predicted):
        if (len(original) == len(predicted)):
            error = 0
            for i in range(len(original)):
                error = error + ((original.iloc[i,:] - predicted.iloc[i,:]) ** 2)
            error = error**.5
            return (error)
        else:
            print("error not computed.")
            return([0,0,0])

   ##############################
   # Statistical methods defined
   ##############################

   #1: AR
    def AutoRegression(self, data, exodata):
        
        #currently, exodata not used.
        #make a dataframe the size of prediction 
        datahat= pd.DataFrame(np.zeros(shape =((self.end - self.start), 3)))
        
        # create a model for each axis and predict each axis
        for i in range(3):
            # make prediction 
            x = data[i].values.tolist()  #get the col values to be a list
            model = AutoReg(x, lags=self.lg)
            model_fit = model.fit()
            datahat.iloc[:,i]= model_fit.predict(self.start, self.end-1)
        return (datahat)
    
    #2 MA
    def MovingAverage(self, data, exodata):

        #make a dataframe the size of prediction 
        datahat= pd.DataFrame(np.zeros(shape =((self.end - self.start), 3)))

        # create a model for each axis and predict each axis
        for i in range(3):
            # make prediction 
            x = data[i].values.tolist()  #get the col values
            model = ARMA(x, order=(0, 1)) #ARMA with a 0,1 input is just MA
            model_fit = model.fit(disp=False)
            datahat.iloc[:,i]= model_fit.predict(self.start, self.end-1)
        return (datahat)
    
    ##3 ARMA
    def AutoRegressinveMovingAverage(self, data, exog):

        #make a dataframe the size of prediction 
        datahat= pd.DataFrame(np.zeros(shape =((self.end - self.start), 3)))

        # create a model for each axis and predict each axis
        for i in range(3):
            # make prediction 
            x = data[i].values.tolist()  #get the col values
            model = ARMA(x, order=(2, 1)) #should this be order 2,1?
            model_fit = model.fit(disp=False)
            datahat.iloc[:,i]= model_fit.predict(self.start, self.end-1)
        return (datahat)
   
    ##4 ARIMA
    def AutoRegressiveIntegeratedMovingAverage(self, data, exog):
        #make a dataframe the size of prediction 
        datahat= pd.DataFrame(np.zeros(shape =((self.end - self.start), 3)))

        # create a model for each axis and predict each axis
        for i in range(3):
            # make prediction 
            x = data[i].values.tolist()  #get the col values
            model = ARIMA(x, order=(1,1,1)) #should this be  1 1 1?
            model_fit = model.fit(disp=False)
            datahat.iloc[:,i]= model_fit.predict(self.start, self.end-1)
        return (datahat)

    #5 SARIMA
    def SeasonalAutoRegressiveIntegratedMovingAverage(self, data, exodata):
        #currently, exodata not used.
        #make a dataframe the size of prediction 
        datahat= pd.DataFrame(np.zeros(shape =((self.end - self.start), 3)))
        
        # create a model for each axis and predict each axis
        for i in range(3):
            # make prediction 
            x = data[i].values.tolist()  #get the col values to be a list
            model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
            model_fit = model.fit(disp=False)
            datahat.iloc[:,i]= model_fit.predict(self.start, self.end-1)
        return (datahat)

    ########NOTE
    ########6, 7, 8 not tested yet....
    ##6 SAIMAER-- needs exogonous data...not sure how we wan to feed this in.
    def SeasonaAutoIntegratedMovingAverageExogenousRegressors(self,data, exodata):
        #currently, exodata not used.
        #make a dataframe the size of prediction 
        datahat= pd.DataFrame(np.zeros(shape =((self.end - self.start), 3)))
        
        # create a model for each axis and predict each axis
        for i in range(3):
            # make prediction 
            x = data[i].values.tolist()  #get the col values to be a list
            x1 = exodata[i].values.tolist()
            model = SARIMAX(x, exog=x1, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
            model_fit = model.fit(disp=False)
            datahat.iloc[:,i]= model_fit.predict(self.start, self.end-1, exog=[exodata]) #not sure here
    
        return (datahat)

    ##7 VAR
    def VectorAutoRegressive(self, data, exodata):
        #currently, exodata not used.
        #make a dataframe the size of prediction 
        datahat= pd.DataFrame(np.zeros(shape =((self.end - self.start), 3)))
        #convert to a list
        datalist = data.values.tolist()
        # create a model for each axis and predict each axis
        model = VAR(datalist)
        model_fit = model.fit()
        datahat = model_fit.forecast(model_fit.y, steps=(self.end -self.start))
        return (datahat)    
    
    ##8 VARMA
    def VectorAutoRegressiveMovingAverage(self):

        #currently, exodata not used.
    
    #make a dataframe the size of prediction 
        datahat= pd.DataFrame(np.zeros(shape =((self.end - self.start), 3)))
        #convert to a list
        datalist = data.values.tolist()
        # create a model for each axis and predict each axis
        model = VARMAX(datalist, order =(1,1))
        model_fit = model.fit(disp=False)
        datahat = model_fit.forecast(model_fit.y, steps=(self.end -self.start))
        return (datahat)


    # 9HWES example

    def HoltWintersExponentialSmoothing(self, data, exodata):
        #currently, exodata not used.
        #make a dataframe the size of prediction 
        datahat= pd.DataFrame(np.zeros(shape =((self.end - self.start), 3)))

        for j in range(len(datahat)):
            # create a model for each axis and predict each axis
            for i in range(3):
                # make prediction 
                x = data[i].values.tolist()  #get the col values to be a list
                #append the predictions made thus far...if any made
                if (j > 0):
                    for k in range (j):
                        x.append(datahat.iloc[k,i])
                       
                #use the original data and predictions to make next prediction
                model = ExponentialSmoothing(x)
                model_fit = model.fit()
                datahat.iloc[j,i]= model_fit.predict(self.end+1, self.end+1) 

        return (datahat)