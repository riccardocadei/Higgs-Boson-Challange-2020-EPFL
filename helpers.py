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
            
def missing_values(X): 
      N, D = X.shape
      missing_data = np.zeros(D)
      delete = [] 
      for feature in range(D):
            missing_data[feature] = np.count_nonzero(X[:,feature]==-999)/N
            if missing_data[feature]>0.3: 
            # delete features with more than 30% missing values
                  delete.append(feature)
            elif missing_data[feature]>0:
            # complete features with less than30% missing values
                  mean = np.where(X[:,feature]==-999, 0, X[:,feature]).mean()
                  X[:,feature] = np.where(X[:,feature]==-999, mean, X[:,feature])
      X = np.delete(X, delete, 1)
      # adding an intercepta
      X = np.c_[np.ones((X.shape[0], 1)), X]

      return X
    
def normalize(X):
      for i in range(x.shape(1)):
            X[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])
      return X
