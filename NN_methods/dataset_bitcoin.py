# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:24:39 2019

@author: parinaz
"""

import numpy as np
import pandas as pd
import random
import math
import os
import requests

def loadData(curr, past_window, future_window):
    """
    Return the historical data for the USD or EUR bitcoin value. Is done with an web API call.
    curr = "USD" | "EUR"
    """
    # For more info on the URL call, it is inspired by :
    # https://github.com/Levino/coindesk-api-node
    r = requests.get(
        "http://api.coindesk.com/v1/bpi/historical/close.json?start=2010-07-17&end=2017-03-03&currency={}".format(
            curr
        )
    )
    data = r.json()
    time_to_values = sorted(data["bpi"].items())
    values = [val for key, val in time_to_values]
    kept_values = values[1000:]
    X = []
    Y = []
    for i in range(len(kept_values) - (future_window + past_window)):
        X.append(kept_values[i:i + past_window])
        Y.append(kept_values[i + past_window:i + (future_window + past_window)])
        
    X = np.array(X)
    Y = np.array(Y)
    # To be able to concat on inner dimension later on:
    X = np.expand_dims(X, axis=2)
    Y = np.expand_dims(Y, axis=2)

    return X, Y

def normalize(X, Y=None):
    """
    Normalise X and Y according to the mean and standard deviation of the X values only.
    """
    # # It would be possible to normalize with last rather than mean, such as:
    # lasts = np.expand_dims(X[:, -1, :], axis=1)
    # assert (lasts[:, :] == X[:, -1, :]).all(), "{}, {}, {}. {}".format(lasts[:, :].shape, X[:, -1, :].shape, lasts[:, :], X[:, -1, :])
    mean = np.expand_dims(np.average(X, axis=1) + 0.00001, axis=1)
    stddev = np.expand_dims(np.std(X, axis=1) + 0.00001, axis=1)
    # print (mean.shape, stddev.shape)
    # print (X.shape, Y.shape)
    X = X - mean
    X = X / (2.5 * stddev)
    if Y is not None:
        #assert Y.shape == X.shape, (Y.shape, X.shape)
        Y = Y - mean
        Y = Y / (2.5 * stddev)
        return X, Y
    return X


def generate_x_y_data(isTrain, batch_size, steps_per_epoch, past_window, future_window):
    """
    Return past and future data for psm1 tool positions.

    We normalize X and Y data according to the X only to not
    spoil the predictions we ask for.

    For every window (window or seq_length), Y is the prediction following X.
    Train and test data are separated according to the 80/20 rule.
    Therefore, the 20 percent of the test data are the data that we should pedict. 
    It is to be noted that the returned X and Y has the same shape
    and are in a tuple.
    """
    # 40 pas values for encoder, 40 after for decoder's predictions.
    #seq_length = 15

    global Y_train
    global X_train
    global X_test
    global Y_test
    # First load, with memoization:
    #if len(Y_test) == 0:
    while True:

        for _ in range(steps_per_epoch):
            # API call:
            X_usd, Y_usd = loadData("USD", past_window, future_window)
            X_eur, Y_eur = loadData("EUR", past_window, future_window)
            # All data, aligned:
            X = np.concatenate((X_usd, X_eur), axis=2)
            Y = np.concatenate((Y_usd, Y_eur), axis=2)
            
            X, Y = normalize(X, Y)
    
            # Split 80-20:
            X_train = X[:int(len(X) * 0.8)]
            Y_train = Y[:int(len(Y) * 0.8)]
            X_Val = X[int(len(X) * 0.8):int(len(X) * 0.9)]
            Y_Val = Y[int(len(Y) * 0.8):int(len(X) * 0.9)]
            X_test = X[int(len(X) * 0.9):]
            Y_test = Y[int(len(X) * 0.9):]
            if isTrain:
                #assert X_train.shape == Y_train.shape, (X_train.shape, Y_train.shape)
                idxes = np.random.randint(X_train.shape[0], size=batch_size)
                X_out = np.array(X_train[idxes]).transpose((1, 0, 2))
                Y_out = np.array(Y_train[idxes]).transpose((1, 0, 2))
                decoder_input =  np.zeros((Y_out.shape[0], Y_out.shape[1], 1))
                #print(X_out.shape)
                yield (X_out, Y_out)
            else:
                #assert X_test.shape == Y_test.shape, (X_test.shape, Y_test.shape)
                idxes = np.random.randint(X_test.shape[0], size=batch_size)
                X_out = np.array(X_test[idxes]).transpose((1, 0, 2))
                Y_out = np.array(Y_test[idxes]).transpose((1, 0, 2))
                decoder_input =  np.zeros((Y_out.shape[0], Y_out.shape[1], 1))
                yield (X_out, Y_out)
