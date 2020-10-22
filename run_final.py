import numpy as np
from helpers import *
from methods import *
from process_data import *
from crossValidation import *
from select_parameter import *

# %load_ext autoreload
# %autoreload 2

seed=20

from zipfile import ZipFile 
  
# # specifying the zip file name 
file_name = 'Data/test.csv.zip'
  
# opening the zip file in READ mode 
with ZipFile(file_name, 'r') as zip: 
    zip.extractall('Data/') 

y, tX, ids = load_csv_data('Data/train.csv')
_, tX_test, ids_test = load_csv_data('Data/test.csv')

# Split data in subsets corresponding to a jet value
msks_jet_train = get_jet_masks(tX)
msks_jet_test = get_jet_masks(tX_test)

# Degree polynomial expansion
degrees = [5,7,6,6]
alpha = 0
# Ridge regression parameters for each subset
lambdas = [0,0,0,0]

# Vector to store the final prediction
y_pred = np.zeros(tX_test.shape[0])

for idx in range(len(msks_jet_train)):
    x_train = tX[msks_jet_train[idx]]
    x_test = tX_test[msks_jet_test[idx]]
    y_train = y[msks_jet_train[idx]]

    # Pre-processing of data
    x_train, x_test = process_data(x_train, x_test, alpha)
    x_train, x_test = phi(x_train, x_test, degrees[idx])

    loss, weights = ridge_regression(y_train, x_train, lambdas[idx])

    y_test_pred = predict_labels(weights, x_test)

    y_pred[msks_jet_test[idx]] = y_test_pred
