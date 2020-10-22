import numpy as np
import random

def process_data(x_train, x_test, alpha=0, add_constant_col=False):
    """
    Impute missing data and compute inverse log values of positive columns
    """
    
    # Consider the 0s in the 'PRI_jet_all_pt' as missing values
    #x_train[:,-1]=np.where(x_train[:,-1]==0, -999, x_train[:,-1])
    
    # Impute missing data
    x_train, x_test = missing_values(x_train, x_test)
    
    x_train[:,[14,17,24,27]]= abs(x_train[:,[14,17,24,27]])
    x_test[:,[14,17,24,27]]= abs(x_test[:,[14,17,24,27]])    
    
    
    # outliers
    x_train = outliers(x_train, alpha)
    x_test = outliers(x_test, alpha)
    
    # other trasformation for positive features
    x_train, x_test = other_transf(x_train, x_test)
    
    x_train, mean_x_train, std_x_train = standardize(x_train)
    x_test, _, _ = standardize(x_test, mean_x_train, std_x_train)
    
    # Delete 5 phi features
    #x_train = np.delete(x_train, [15,16,18,20,25,28], 1)
    #x_test = np.delete(x_test, [15,16,18,20,25,28], 1)
    
    x_train = np.delete(x_train, [15,16,18,20], 1)
    x_test = np.delete(x_test, [15,16,18,20], 1)
    
    # Add an intercepta
    if add_constant_col is True:
        x_train = add_constant_column(x_train)
        x_test = add_constant_column(x_test)
        
    return x_train, x_test


def phi(x_train, x_test, degree):
    
    N, D = x_train.shape
    
    # Polynomial expansion
    x_train = build_poly2(x_train, degree)
    x_test = build_poly2(x_test, degree)
    
    return x_train, x_test


###############################################################


def Random_Over_Sampling(tX, y):
    
        # Class count
        count_class_0 = np.count_nonzero(y==-1)
        count_class_1 = np.count_nonzero(y==1)

        # Divide by class
        class_0 = tX[np.where(y==-1)]
        class_1 = tX[np.where(y==1)]
        
        #count_class_1_over = count_class_0-count_class_1
        count_class_1_over = int((count_class_0-count_class_1)/3)
        class_1_over_indx = random.sample(set(np.arange(count_class_1)), count_class_1_over)
        class_1_over = class_1[class_1_over_indx]

        tX = np.concatenate((tX, class_1_over))
        y = np.concatenate((y, np.ones(count_class_1_over)))

        new_ord = np.random.permutation(y.shape[0])
        tX = tX[new_ord]
        y = y[new_ord]
        
        return tX, y
    
    
def missing_values(X, X_test): 
        N, D = X.shape
        missing_data = np.zeros(D)
        missing_cols = [] 
        for feature in range(D):
            missing_data[feature] = np.count_nonzero(X[:,feature]==-999)/N
            
            # delete features with more than 80% missing values
            if missing_data[feature]>0.8: 
                    missing_cols.append(feature)
            
            # complete features with less than 80% missing values
            elif missing_data[feature]>0:
                    X_feature = X[:,feature]
                    X_feature = X_feature[X_feature!=-999]
                    values, counts = np.unique(X_feature, return_counts = True)
                    moda = values[np.argmax(counts)]
                    X[:,feature] = np.where(X[:,feature]==-999, moda, X[:,feature])
                    X_test[:,feature] = np.where(X_test[:,feature]==-999, moda, X_test[:,feature])
        X[:,missing_cols]=0
        X_test[:,missing_cols]=0
        
        return X, X_test
    

def outliers(x, alpha=0):
    for i in range(x.shape[1]):
        x[:,i][ x[:,i]<np.percentile(x[:,i],alpha) ] = np.percentile(x[:,i],alpha)
        x[:,i][ x[:,i]>np.percentile(x[:,i],100-alpha) ] = np.percentile(x[:,i],100-alpha)
        
    return x


# Other traformation 
def other_transf(x_train, x_test):
    
    # Create inverse log values of features which are positive in value
    inv_log_cols = [0, 1, 2, 5, 7, 9, 10, 13, 16, 19, 21, 23, 26]
    
    x_train_t1 = np.log(1 / (1 + x_train[:, inv_log_cols]))
    x_train = np.hstack((x_train, x_train_t1))
    
    x_test_t1 = np.log(1 / (1 + x_test[:, inv_log_cols]))
    x_test = np.hstack((x_test, x_test_t1))
    
    return x_train, x_test  

    
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


    


def build_poly3(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree + couplings and root square"""
    # ATTENTION: np.c_() is super expensive!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    N, D = x.shape    
    
    # couples
    temp_dict2 = {}
    count2 = 0
    for i in range(D):
        for j in range(i+1,D):
            temp = x[:,i] * x[:,j]
            temp_dict2[count2] = [temp]
            count2 += 1
            
    # triples
    temp_dict3 = {}
    count3 = 0
    for i in range(D):
        for j in range(i,D):
            for k in range(j,D):
                if i!=j or j!=k:
                    temp = x[:,i]*x[:,j]*x[:,k]
                    temp_dict3[count3] = [temp]
                    count3 += 1
    
        
    poly = np.zeros(shape = (N, 1+D*(degree+1)+count2+count3+1))
    
    # intercept
    poly[:,0] = np.ones(N)
    
    # powers
    for deg in range(1,degree+1):
        for i in range(D):
            poly[:, 1+D*(deg-1)+i ] = np.power(x[:,i],deg)
    
    # coupling     
    for i in range(count2):
        poly[:, 1+D*degree+i ] = temp_dict2[i][0]
        
    # triples     
    for i in range(count3):
        poly[:, 1+D*degree+count2+i ] = temp_dict3[i][0]
            
    # roots   
    for i in range(D):
        poly[:, 1+D*degree+count2+count3+i] = np.abs(x[:,i])**0.5
    

    return poly

def rad(x,t):
    N, D = x.shape
    r = np.zeros([N,D])
    for i in range(N):
        for j in range(D):
            if x[i,j]>0:
                r[i,j] = x[i,j]**(1/t)
            else:
                r[i,j] = -(-x[i,j])**(1/t)
    return r 

def build_poly2(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree + couplings and root square"""
    # ATTENTION: np.c_() is super expensive!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    N, D = x.shape    
    
    # couples
    temp_dict2 = {}
    count2 = 0
    for i in range(D):
        for j in range(i+1,D):
            temp = x[:,i] * x[:,j]
            temp_dict2[count2] = [temp]
            count2 += 1
    
    poly = np.zeros(shape = (N, 1+D*(degree+2)+count2+1))
    
    # intercept
    poly[:,0] = np.ones(N)
    
    # powers
    for deg in range(1,degree+1):
        for i in range(D):
            poly[:, 1+D*(deg-1)+i ] = np.power(x[:,i],deg)
    
    # coupling     
    for i in range(count2):
        poly[:, 1+D*degree+i ] = temp_dict2[i][0]
 
            
    # roots   
    for i in range(D):
        poly[:, 1+D*degree+count2+i] = np.abs(x[:,i])**0.5
        
    poly[:, 1+D*degree+count2+D:-1] = rad(x, 3)
    

    return poly
 

   
    





    
        


def get_jet_masks(x):
    """
    Returns 3 masks corresponding to the rows of x with a jet value
    of 0, 1 and  2 or 3 respectively.
    """
    return {
        0: x[:, 22] == 0,
        1: x[:, 22] == 1,
        2: np.logical_or(x[:, 22] == 2, x[:, 22] == 3)
        #2: x[:, 22] == 2, 
        #3: x[:, 22] == 3
    }



# unfortunately too expensive
def impute(x,nu):
    
    remember=x[:,22]
    N,D = x.shape
    idx = get_jet_masks(x)
    
    x, x = missing_values(x, x)
    x,_,_ = standardize (x)
    
    cols = set(range(D))
    
    # class 1
    col1 = set([4,5,6,12,26,27,28])
    col1n = cols-col1
    
    idx23 = np.array(idx[2])+np.array(idx[3])
    
    x1 = x[idx[1],:]
    x23 = x[idx23,:]
    for j in col1:
        for i in range(x[idx[1]].shape[0]):
                key = random.sample(range(x23.shape[0]), nu)
                k = np.argmin(abs((x23[key,:][:,list(col1n)]-x[i,list(col1n)])).sum(axis=1))
                x1[i,j]= x23[key,:][k,j]
    x[idx[1],:] = x1
    
    # class 0
    col0= set([23,24,25,29]).union(col1)
    col0n = cols-col0
    idx123 = np.array(idx[1])+np.array(idx[2])+np.array(idx[3])
 
    x0=x[idx[0],:]
    x123=x[idx123,:]
    
    for j in col0:
        for i in range(x[idx[1]].shape[0]):
                key = random.sample(range(x123.shape[0]), nu)
                k = np.argmin(abs((x123[key,:][:,list(col0n)]-x[i,list(col0n)])).sum(axis=1))
                x0[i,j]= x123[key,:][k,j]
    x[idx[0],:] = x0
    
    x[:,22]=remember
    
    return x