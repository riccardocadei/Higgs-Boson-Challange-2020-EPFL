import numpy as np
import matplotlib.pyplot as plt
from methods import *
from helpers import *
from process_data import *
#import pandas as pd

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


###########


def cross_validation(y, x, fun, k_indices, k, max_iters, gamma, degrees, batch_size):
    """
    Completes k-fold cross-validation using the least squares method
    solved by Gradient Descent.
    """
    # get k'th subgroup in test, others in train
    msk_test = k_indices[k]
    msk_train = np.delete(k_indices, (k), axis=0).ravel()

    x_train_all_jets = x[msk_train, :]
    x_test_all_jets = x[msk_test, :]
    y_train_all_jets = y[msk_train]
    y_test_all_jets = y[msk_test]

    # split in 4 subsets the training set
    msk_jets_train = get_jet_masks(x_train_all_jets)
    msk_jets_test = get_jet_masks(x_test_all_jets)

    # initialize output vectors
    y_train_pred = np.zeros(len(y_train_all_jets))
    y_test_pred = np.zeros(len(y_test_all_jets))

    for idx in range(len(msk_jets_train)):
        x_train = x_train_all_jets[msk_jets_train[idx]]
        x_test = x_test_all_jets[msk_jets_test[idx]]
        y_train = y_train_all_jets[msk_jets_train[idx]]

        # data pre-processing
        x_train, x_test = process_data(x_train, x_test)
        # transformation
        x_train, x_test = phi(x_train, x_test, degrees[idx])

        # compute weights using given method
        loss, weights = fun(y_train, x_train, None, batch_size, max_iters, gamma)
        
        y_train_pred[msk_jets_train[idx]] = predict_labels_logistic(weights, x_train)
        y_test_pred[msk_jets_test[idx]] = predict_labels_logistic(weights, x_test)

    # compute accuracy for train and test data
    acc_train = compute_accuracy(y_train_pred, y_train_all_jets)
    acc_test = compute_accuracy(y_test_pred, y_test_all_jets)

    return acc_train, acc_test



### MODEL 1

def cross_validation_least_squares_GD(y, x, k_indices, k, max_iters, gamma, degrees):
    """
    Completes k-fold cross-validation using the least squares method
    solved by Gradient Descent.
    """
    # get k'th subgroup in test, others in train
    msk_test = k_indices[k]
    msk_train = np.delete(k_indices, (k), axis=0).ravel()

    x_train_all_jets = x[msk_train, :]
    x_test_all_jets = x[msk_test, :]
    y_train_all_jets = y[msk_train]
    y_test_all_jets = y[msk_test]

    # split in 4 subsets the training set
    msk_jets_train = get_jet_masks(x_train_all_jets)
    msk_jets_test = get_jet_masks(x_test_all_jets)

    # initialize output vectors
    y_train_pred = np.zeros(len(y_train_all_jets))
    y_test_pred = np.zeros(len(y_test_all_jets))

    for idx in range(len(msk_jets_train)):
        x_train = x_train_all_jets[msk_jets_train[idx]]
        x_test = x_test_all_jets[msk_jets_test[idx]]
        y_train = y_train_all_jets[msk_jets_train[idx]]

        # data pre-processing
        x_train, x_test = process_data(x_train, x_test)
        # transformation
        x_train, x_test = phi(x_train, x_test, degrees[idx])

        # compute weights using given method
        weights, loss = least_squares_GD(y_train, x_train, None, max_iters, gamma)
        
        y_train_pred[msk_jets_train[idx]] = predict_labels(weights, x_train)
        y_test_pred[msk_jets_test[idx]] = predict_labels(weights, x_test)

    # compute accuracy for train and test data
    acc_train = compute_accuracy(y_train_pred, y_train_all_jets)
    acc_test = compute_accuracy(y_test_pred, y_test_all_jets)

    return acc_train, acc_test


#### MODEL 3

def cross_validation_least_squares(y, x, k_indices, k, degrees, alpha):
    """
    Completes k-fold cross-validation using the least squares method
    solved by Normal Equations.
    """
    
    # Random Over Sampling
    #x,y = Random_Over_Sampling(x,y)
    
    # get k'th subgroup in test, others in train
    msk_test = k_indices[k]
    msk_train = np.delete(k_indices, (k), axis=0).ravel()

    x_train_all_jets = x[msk_train, :]
    x_test_all_jets = x[msk_test, :]
    y_train_all_jets = y[msk_train]
    y_test_all_jets = y[msk_test]

    # split in 4 subsets the training set
    msk_jets_train = get_jet_masks(x_train_all_jets)
    msk_jets_test = get_jet_masks(x_test_all_jets)

    # initialize output vectors
    y_train_pred = np.zeros(len(y_train_all_jets))
    y_test_pred = np.zeros(len(y_test_all_jets))

    for idx in range(len(msk_jets_train)):
        x_train = x_train_all_jets[msk_jets_train[idx]]
        x_test = x_test_all_jets[msk_jets_test[idx]]
        y_train = y_train_all_jets[msk_jets_train[idx]]

        if len(x_train!=0):
            # data pre-processing
            x_train, x_test = process_data(x_train, x_test, alpha)
            # transformation
            x_train, x_test = phi(x_train, x_test, degrees[idx])
            # compute weights using given method
            weights, loss  = least_squares(y_train, x_train)

            y_train_pred[msk_jets_train[idx]] = predict_labels(weights, x_train)
            y_test_pred[msk_jets_test[idx]] = predict_labels(weights, x_test)

    # compute accuracy for train and test data
    acc_train = compute_accuracy(y_train_pred, y_train_all_jets)
    acc_test = compute_accuracy(y_test_pred, y_test_all_jets)

    return acc_train, acc_test

##### MODEL 4

def cross_validation_ridge_regression(y, x, k_indices, k, lambdas, degrees, alpha):
    """
    Completes k-fold cross-validation using the ridge regression method.
    """
    # get k'th subgroup in test, others in train
    msk_test = k_indices[k]
    msk_train = np.delete(k_indices, (k), axis=0).ravel()

    x_train_all_jets = x[msk_train, :]
    x_test_all_jets = x[msk_test, :]
    y_train_all_jets = y[msk_train]
    y_test_all_jets = y[msk_test]

    # split in 4 subsets the training set
    msk_jets_train = get_jet_masks(x_train_all_jets)
    msk_jets_test = get_jet_masks(x_test_all_jets)

    # initialize output vectors
    y_train_pred = np.zeros(len(y_train_all_jets))
    y_test_pred = np.zeros(len(y_test_all_jets))

    for idx in range(len(msk_jets_train)):
        x_train = x_train_all_jets[msk_jets_train[idx]]
        x_test = x_test_all_jets[msk_jets_test[idx]]
        y_train = y_train_all_jets[msk_jets_train[idx]]

        if len(x_train!=0):
            # data pre-processing
            x_train, x_test = process_data(x_train, x_test, alpha)
            # transformation
            x_train, x_test = phi(x_train, x_test, degrees[idx])

            # compute weights using given method
            weights,loss = ridge_regression(y=y_train, tx=x_train, lambda_=lambdas[idx])
            
            y_train_pred[msk_jets_train[idx]] = predict_labels(weights, x_train)
            y_test_pred[msk_jets_test[idx]] = predict_labels(weights, x_test)

    # compute accuracy for train and test data
    acc_train = compute_accuracy(y_train_pred, y_train_all_jets)
    acc_test = compute_accuracy(y_test_pred, y_test_all_jets)

    return acc_train, acc_test












