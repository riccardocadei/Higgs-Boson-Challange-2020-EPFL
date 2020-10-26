import numpy as np
from helpers import *
from implementations import *
from process_data import *
from crossvalidation import *

seed=10


from zipfile import ZipFile 
  
# Specifying the zip file name 
file_name = 'Data/test.csv.zip'
  
# Opening the zip file in READ mode 
with ZipFile(file_name, 'r') as zip: 
    zip.extractall('Data/') 

    
    
# 1. Load the dataset 
y, tX, ids = load_csv_data('Data/train.csv')
_, tX_test, ids_test = load_csv_data('Data/test.csv')

# Inizialize a vector to store the final prediction
y_pred = np.zeros(tX_test.shape[0])


# 2. Split the training set and test set in subsets according to the jet value
msks_jet_train = get_jet_masks(tX)
msks_jet_test = get_jet_masks(tX_test)


# 3. Set the parameters 
# found using Grid Search to optimize the accuracy predicted through Cross Validation
# Preprocessing parameters


# Coefficients for outliers detection and cutting for each subset
alphas = [4, 4, 5]
# Degree of polynomial expansion for each subset
degrees = [5, 5, 5]
# Ridge regression lambda parameter for each subset
lambdas = [1e-06, 1e-05, 1e-03]


# 4. For each subset train the model and make prediction 
for idx in range(len(msks_jet_train)):
    x_train = tX[msks_jet_train[idx]]
    x_test = tX_test[msks_jet_test[idx]]
    y_train = y[msks_jet_train[idx]]

    # Pre-processing and transformation of the training set and test set
    x_train, x_test = process_data(x_train, x_test, alphas[idx])
    x_train, x_test = phi(x_train, x_test, degrees[idx])
    
    # Train the model through Ridge Regression
    weights, _ = ridge_regression(y_train, x_train, lambdas[idx])
    
    # Prediction
    y_test_pred = predict_labels(weights, x_test)
    y_pred[msks_jet_test[idx]] = y_test_pred

    
# 5. Submission
OUTPUT_PATH = 'data/finalsubmission.csv' 
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
