# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
import matplotlib.pyplot as plt

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
            
# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from methods import *

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
    for i in range(x_train.shape[1]):
            tX_i = x_train[:,i]
            if tX_i[tX_i>0].shape[0] == x_train.shape[0]:
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

# Cross Validation

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, y_tr, x_te, y_te

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_):
    """return the loss of ridge regression."""
    # *************************************************
    y_test = y[k_indices[k]]
    x_test = x[k_indices[k]]
    y_train = np.delete(y,k_indices[k])
    x_train =  np.delete(x,k_indices[k],axis=0)
    _,w = ridge_regression(y_train, x_train, lambda_)
    loss_te = compute_mse(y_test, x_test, w)
    loss_tr = compute_mse(y_train, x_train, w)
    return loss_tr, loss_te, w

def cross_validation_demo(y,x):
    seed = 10
    k_fold = 10
    lambdas = np.logspace(-4, -1, 100)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    # INSERT YOUR CODE HERE
    # cross validation: TODO
    # ***************************************************    
    for indexl, l in enumerate(lambdas):
        print(indexl)
        tr = 0
        te = 0
        for k in range(k_fold):
            tr_temp, te_temp, w = cross_validation(y, x, k_indices, k, l)
            tr = tr + tr_temp
            te = te + te_temp
        rmse_tr.append(tr)
        rmse_te.append(te)
        if te == np.min(rmse_te): 
            best_lambda = l 
            best_weights = w
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    
    return tr,best_weights, best_lambda


def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
