
import numpy as np
import requests
import random
import math


def loadCurrency(curr, past_window, future_window):
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
    for i in range(len(kept_values) - (past_window + future_window)):
        X.append(kept_values[i:i + past_window])
        Y.append(kept_values[i + past_window:i + past_window + future_window])

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


def fetch_batch_size_random(X, Y, batch_size):
    """
    Returns randomly an aligned batch_size of X and Y among all examples.
    The external dimension of X and Y must be the batch size (eg: 1 column = 1 example).
    X and Y can be N-dimensional.
    """
    #assert X.shape == Y.shape, (X.shape, Y.shape)
    idxes = np.random.randint(X.shape[0], size=batch_size)
    X_out = np.array(X[idxes]).transpose((1, 0, 2))
    Y_out = np.array(Y[idxes]).transpose((1, 0, 2))
    return X_out, Y_out

X_train = []
Y_train = []
X_test = []
Y_test = []


def generate_x_y_data(Status, batch_size, past_window, future_window):
    """
    Return financial data for the bitcoin.

    Features are USD and EUR, in the internal dimension.
    We normalize X and Y data according to the X only to not
    spoil the predictions we ask for.

    For every window (window or seq_length), Y is the prediction following X.
    Train and test data are separated according to the 80/20 rule.
    Therefore, the 20 percent of the test data are the most
    recent historical bitcoin values. Every example in X contains
    40 points of USD and then EUR data in the feature axis/dimension.
    It is to be noted that the returned X and Y has the same shape
    and are in a tuple.
    """
    # 40 pas values for encoder, 40 after for decoder's predictions.
    seq_length = 7

    global Y_train
    global X_train
    global X_test
    global Y_test
    global X_Val
    global Y_Val
    # First load, with memoization:
    if len(Y_test) == 0:
        # API call:
        X_usd, Y_usd = loadCurrency("USD", past_window = past_window, future_window=future_window)
        X_eur, Y_eur = loadCurrency("EUR", past_window, future_window)

        # All data, aligned:
        X = np.concatenate((X_usd, X_eur), axis=2)
        Y = np.concatenate((Y_usd, Y_eur), axis=2)
        X, Y = normalize(X, Y)

        # Split 80-10-10:
        X_train = X[:int(len(X) * 0.8)]
        Y_train = Y[:int(len(Y) * 0.8)]
        X_Val = X[int(len(X) * 0.8):int(len(X) * 0.9)]
        Y_Val = Y[int(len(Y) * 0.8):int(len(X) * 0.9)]
        X_test = X[int(len(X) * 0.9):]
        Y_test =  X[int(len(X) * 0.9):]
    if Status == "Train":
        return fetch_batch_size_random(X_train, Y_train, batch_size)
    if Status == "Validation":
        return fetch_batch_size_random(X_Val,  Y_Val,  batch_size)
    if Status == "Test":
        return fetch_batch_size_random(X_test, Y_test, batch_size)
