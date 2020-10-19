
# Useful starting lines
import numpy as np
from helpers import *
from methods import *
from process_data import *
from crossvalidation import *

# %load_ext autoreload
# %autoreload 2

seed=20

from zipfile import ZipFile 
  
# # specifying the zip file name 
file_name = 'Data/test.csv.zip'
  
# opening the zip file in READ mode 
with ZipFile(file_name, 'r') as zip: 
    zip.extractall('Data/') 

#load the dataset 
y, tx, ids = load_csv_data('Data/train.csv')
_, tx_test, ids_test = load_csv_data('Data/test.csv.zip')

#preprocessing

# higgs = np.count_nonzero(y==1)
# print(f'From {y.shape[0]} training examples, {higgs} are 1, i.e. the {higgs/y.shape[0]} %')

#manage mising values by deleting or ceompleting  feature 
tx, tx_test = missing_values(tx, tx_test)

#standardization
tx, mean_tx, std_tx = standardize(tx)
tx_test, _, _ = standardize(tx, mean_tx, std_tx)

#argumenting extention with inverse log values
tx, tx_test = process_data(tx, tx_test, True)

#1. Least Squares with Gradient Descent
loss, weights = least_squares_GD(y, tx, initial_w=np.zeros(tx.shape[1]), max_iters=1000, gamma=0.005)

#2. Least Squares with Stochastic Gradient Descent
loss, weights = least_squares_SGD(y, tx, initial_w=np.zeros(tx.shape[1]), batch_size=1, max_iters=1000, gamma=0.005)

#3. Least Squares with Normal Equations
loss, weights = least_squares(y, tx)

#4. Ridge regression with Normal Equations
loss, weights = ridge_regression(y, tx, 0.2)

# 5. Logistic Regression with Stochastic Gradient Descent
loss, weights = logistic_regression(y, tX,initial_w= np.random.random(tX.shape[1]))

# 6. Regularized Logistic Regression with Stochastic Gradient Descent
# ***************************************************
# loss, weights = reg_logistic regression_SGD(y, tX, initial w, max_iters, gamma)
# TODO
# ***************************************************

#Cross Validation
loss, weights, best_lambda = cross_validation_ridge_regression(y,tx)

#Submission
OUTPUT_PATH = 'Data/firstsubmission.csv' 
y_pred = predict_labels(weights, tx_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
