import numpy as np
import matplotlib.pyplot as plt
from methods import *
from helpers import *
from process_data import *
from crossValidation import *

def best_degree_lamb_selection(degrees, lambdas, k_fold, y, tx, alpha, seed):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    comparison = []

    for deg in degrees:
        # cross validation
        degree = [deg]*4
        for lamb in lambdas:
            lambda_ = [lamb]*4
            accs_test = []
            for k in range(k_fold):
              _, acc_test = cross_validation_ridge_regression(y, tx, k_indices, k, lambda_, degree, alpha)
              accs_test.append(acc_test)
            comparison.append([deg,lamb,np.mean(accs_test)])
    
    comparison = np.array(comparison)
    ind_best =  np.argmax(comparison[:,2])      
    best_degree = comparison[ind_best,0]
    best_lamb = comparison[ind_best,1]
    accu = comparison[ind_best,2]
   
    return best_degree, best_lamb, accu


def select_parameters_ridge_regression(y,tX,degrees,lambdas,alpha,k_fold,seed):  
    par_degree = []
    par_lamb = []
    accus = []

    # split in 4 subsets the training set
    msk_jets = get_jet_masks(tX)

    for idx in range(len(msk_jets)):
        tx = tX[msk_jets[idx]]
        ty = y[msk_jets[idx]]
        
        degree,lamb,accu = best_degree_lamb_selection(degrees, lambdas, k_fold, ty, tx, alpha,seed)
        par_degree.append(degree)
        par_lamb.append(lamb)
        accus.append(accu)

    return par_degree, par_lamb, accus

""" for least square """
def best_degree_selection(degrees, k_fold, y, tx, alpha, seed):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    comparison = []

    for deg in degrees:
        # cross validation
        degree = [deg]*4
        accs_test = []
        for k in range(k_fold):
          _, acc_test = cross_validation_least_squares(y, tx, k_indices, k, degrees, alpha)
          accs_test.append(acc_test)
        comparison.append([deg,np.mean(accs_test)])
    
    comparison = np.array(comparison)
    ind_best =  np.argmax(comparison[:,1])      
    best_degree = comparison[ind_best,0]
    accu = comparison[ind_best,1]
   
    return best_degree, accu
    
def select_parameters_least_squares(y,tX,degrees,alpha,k_fold,seed):  
    par_degree = []
    accus = []

    # split in 4 subsets the training set
    msk_jets = get_jet_masks(tX)

    for idx in range(len(msk_jets)):
        tx = tX[msk_jets[idx]]
        ty = y[msk_jets[idx]]
        
        degree, accu = best_degree_selection(degrees, k_fold, ty, tx, alpha,seed)
        par_degree.append(degree)
        accus.append(accu)

    return par_degree,  accus