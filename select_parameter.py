import numpy as np
from implementations import *
from helpers import *
from process_data import *
from crossvalidation import *

def select_parameters_ridge_regression_jet(y,tX,degrees,lambdas,alphas,k_fold,seed):
    """
    Given the training set and a set of tuples of parameters (alphas, lamdas, degrees) 
    for each jet_subset returns the tuple which maximize the accuracy predicted through Cross Validation 
    """
    par_degree = []
    par_lamb = []
    par_alpha = []
    accus = []

    # Split the training set in subsets according to the jet value 
    msk_jets = get_jet_masks(tX)

    for idx in range(len(msk_jets)):
        tx = tX[msk_jets[idx]]
        ty = y[msk_jets[idx]]
        
        degree,lamb,alpha,accu = select_parameters_ridge_regression(degrees, lambdas, alphas, k_fold, ty, tx, seed)
        par_degree.append(degree)
        par_lamb.append(lamb)
        par_alpha.append(alpha)
        accus.append(accu)

    return par_degree, par_lamb, par_alpha, accus


def select_parameters_ridge_regression(degrees, lambdas, alphas, k_fold, y, tx, seed):
    """
    Given the training set and a set of tuples of parameters (alphas, lamdas, degrees) 
    returns the tuple which maximize the accuracy predicted through Cross Validation 
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    comparison = []

    for degree in degrees:
        for lamb in lambdas:
            for alpha in alphas:
                accs_test = []
                for k in range(k_fold):
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
