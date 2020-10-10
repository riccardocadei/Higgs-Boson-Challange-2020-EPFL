import numpy as np

def missing_values(X, X_test): 
        N, D = X.shape
        missing_data = np.zeros(D)
        missing_cols = [] 
        for feature in range(D):
            missing_data[feature] = np.count_nonzero(X[:,feature]==-999)/N
            if missing_data[feature]>0.8: 
            # delete features with more than 80% missing values
                  missing_cols.append(feature)
            elif missing_data[feature]>0:
            # complete features with less than 80% missing values
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
    # x_train, x_test = missing_values(x_train, x_test)
    
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
