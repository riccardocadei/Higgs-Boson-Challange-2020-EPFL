import numpy as np
import matplotlib.pyplot as plt
from methods import *

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
    y_test = y[k_indices[k]]
    x_test = x[k_indices[k]]
    y_train = np.delete(y,k_indices[k])
    x_train =  np.delete(x,k_indices[k],axis=0)
    loss_tr ,w = ridge_regression(y_train, x_train, lambda_)
    loss_te = compute_mse(y_test, x_test, w)
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