# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
def missing_values(X, X_test): 
        N, D = X.shape
        missing_data = np.zeros(D)
        missing_cols = [] 
        for feature in range(D):
            missing_data[feature] = np.count_nonzero(X[:,feature]==-999)/N
            if missing_data[feature]>0.8: 
            # delete features with more than 30% missing values
                  missing_cols.append(feature)
            elif missing_data[feature]>0:
            # complete features with less than 30% missing values
                  X_feature = X[:,feature]
                  median = np.median(X_feature[X_feature != -999])
                  X[:,feature] = np.where(X[:,feature]==-999, median, X[:,feature])
                  X_test[:,feature] = np.where(X_test[:,feature]==-999, median, X_test[:,feature])
        X = np.delete(X, missing_cols, 1)
        X_test = np.delete(X_test, missing_cols, 1)
      

        return X, X_test
    
def standardize(x, mean_x=None, std_x=None):
    """ Standardize the original data set. """
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x > 0] = x[:, std_x > 0] / std_x[std_x > 0]

    return x, mean_x, std_x  

def add_constant_column(x):
    """ Prepend a column of 1 to the matrix. """
    return np.hstack((np.ones((x.shape[0], 1)), x))


def process_data(x_train, x_test, add_constant_col=True):
    """
    Impute missing data and compute inverse log values of positive columns
    """
    # Impute missing data
    x_train, x_test = missing_values(x_train, x_test)
    
    inv_log_cols=[]
    for i in range(tX.shape[1]):
            tX_i = tX[:,i]
            if tX_i[tX_i>0].shape[0] == tX.shape[0]:
                    inv_log_cols.append(i)

    # Create inverse log values of features which are positive in value.
    x_train_inv_log_cols = np.log(1 / (1 + x_train[:, inv_log_cols]))
    x_train = np.hstack((x_train, x_train_inv_log_cols))

    x_test_inv_log_cols = np.log(1 / (1 + x_test[:, inv_log_cols]))
    x_test = np.hstack((x_test, x_test_inv_log_cols))

    x_train, mean_x_train, std_x_train = standardize(x_train)
    x_test, mean_x_test, std_x_test = standardize(x_test, mean_x_train, std_x_train)

    if add_constant_col is True:
        x_train = add_constant_column(x_train)
        x_test = add_constant_column(x_test)

    return x_train, x_test

# try to use median, avoid this function
def impute_values(x_train, x_test):
    """ Replace missing values (NA) by the most frequent value of the column. """
    for i in range(x_train.shape[1]):
        # If NA values in column
        if na(x_train[:, i]):
            msk_train = (x_train[:, i] != -999.)
            msk_test = (x_test[:, i] != -999.)
            # Replace NA values with most frequent value
            values, counts = np.unique(x_train[msk_train, i], return_counts=True)
            # If there are values different from NA
            if (len(values) > 1):
                x_train[~msk_train, i] = values[np.argmax(counts)]
                x_test[~msk_test, i] = values[np.argmax(counts)]
            else:
                x_train[~msk_train, i] = 0
                x_test[~msk_test, i] = 0

    return x_train, x_test

def na(x):
    """ Identifies missing values. """
    return np.any(x == -999)