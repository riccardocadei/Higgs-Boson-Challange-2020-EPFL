import numpy as np
import matplotlib.pyplot as plt
from methods import *
from helpers import *
from process_data import *
from crossvalidation import *

def best_degree_lamb_selection(degrees, lambdas, alphas, k_fold, y, tx, seed):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    comparison = []

    for degree in degrees:
        for lamb in lambdas:
            for alpha in alphas:
                accs_test = []
                for k in range(k_fold):
                        print(degree,lamb,alpha)
                        _, acc_test = cross_validation(y, tx, ridge_regression, k_indices, k, degree, alpha, lamb)
                        accs_test.append(acc_test)
                        comparison.append([degree,lamb,alpha,np.mean(accs_test)])
    
    comparison = np.array(comparison)
    ind_best =  np.argmax(comparison[:,3])      
    best_degree = comparison[ind_best,0]
    best_lamb = comparison[ind_best,1]
    best_alpha = comparison[ind_best,2]
    accu = comparison[ind_best,3]
   
    return best_degree, best_lamb, best_alpha, accu


def select_parameters_ridge_regression(y,tX,degrees,lambdas,alphas,k_fold,seed):  
    par_degree = []
    par_lamb = []
    par_alpha = []
    accus = []

    # split in 4 subsets the training set
    msk_jets = get_jet_masks(tX)

    for idx in range(len(msk_jets)):
        tx = tX[msk_jets[idx]]
        ty = y[msk_jets[idx]]
        
        degree,lamb,alpha,accu = best_degree_lamb_selection(degrees, lambdas, alphas, k_fold, ty, tx, seed)
        par_degree.append(degree)
        par_lamb.append(lamb)
        par_alpha.append(alpha)
        accus.append(accu)

    return par_degree, par_lamb, par_alpha, accus

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