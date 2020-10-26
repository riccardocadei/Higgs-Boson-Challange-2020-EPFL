import numpy as np

# Gradient based methods for linear systems

def compute_mse(y, tx, w):
    """
    Compute the Mean Square Error as defined in class.
    Takes as input the targeted y, the sample matrix X and the feature fector w.
    """
    e = y - tx@w
    mse = e.T.dot(e) /(2*len(e))
    return mse

def least_squares(y, tx):
    """
    Compute an esimated solution of the problem y = tx @ w, and the associated error. This method is equivalent 
    to the minimization problem of finding w such that |y-tx@w||^2 is minimal. Note that this methods provides the global optimum.
    The error is the mean square error of the targeted y and the solution produced by the least square function.
    Takes as input the targeted y, and the sample matrix X.
    """
    w = np.linalg.solve (tx.T.dot(tx),tx.T.dot(y))
    mse = compute_mse(y, tx, w)
    return w, mse 

def least_squares_gradient(y, tx, w):
    """
    Compute the gradient of the mean square error with respect to w, and the current error vector e.
    Takes as input the targeted y, the sample matrix w and the feature vector w. 
    This function is used when solving gradient based method, such that least_squares_GD() and least_squares_SGD().
    """  
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)
    return grad, e

def least_squares_GD(y, tx, initial_w=None, max_iters=50, gamma=0.1):
    """
    Compute an estimated solution of the problem y = tx @ w and the associated error using Gradient Descent. 
    This method is equivalent to the minimization problem of finding w such that |y-tx@w||^2 is minimal. Note that 
    this method may output a local minimum, while least_squares() provides the global minimum.
    Takes as input:
        * the targeted y
        * the sample matrix w
        * the initial guess for w, by default set as a vector of zeros
        * the number of iterations for Gradient Descent
        * the learning rate gamma
    """
    # Define parameters to store w and loss
    if np.all(initial_w == None): initial_w = np.zeros(tx.shape[1])  
    ws = [initial_w] # Initial guess w0 generated randomly
    losses = []
    w = ws[0]
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = least_squares_gradient(y, tx, w)
        loss = compute_mse(y,tx,w)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        #if (n_iter % int(max_iters/5)) == 0:
            #print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters,l=loss))
    return w,loss

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """ 
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def least_squares_SGD(y, tx, initial_w=None, batch_size=1, max_iters=50, gamma=0.00005):
    """
    Compute an estimated solution of the problem y = tx @ w and the associated error using Stochastic Gradient Descent. 
    Takes as input:
        * the targeted y
        * the sample matrix w
        * the initial guess for w, by default set as a vector of zeros
        * the batch_size, which is the number of samples on which the new gradient is computed. If set to 1 it corresponds
        to Stochastic Gradient Descent, to the full number of samples it is identifical to least_squares_GD().
        * the number of iterations for Gradient Descent
        * the learning rate gamma
    """
    # Define parameters to store w and loss
    if np.all(initial_w == None): initial_w = np.zeros(tx.shape[1])
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = least_squares_gradient(y_batch, tx_batch, np.array(w))
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_mse(y, tx, w)
            # store w and loss
            losses.append(loss)

        #if n_iter % int(max_iters/5) == 0:
            #print("SGD({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
    return w,loss
# RIDGE REGRESSION

def ridge_regression(y, tx, lambda_):
    """
    Compute an esimated solution of the problem y = tx @ w , and the associated error. Note that this method
    is a variant of least_square() but with an added regularization term lambda_. 
    This method is equivalent to the minimization problem of finding w such that |y-tx@w||^2 + lambda_*||w||^2 is minimal. 
    The error is the mean square error of the targeted y and the solution produced by the least square function.
    Takes as input the targeted y, the sample matrix X and the regulariation term lambda_.
    """
    x_t = tx.T
    lambd = lambda_ * 2 * len(y)
    w = np.linalg.solve (np.dot(x_t, tx) + lambd * np.eye(tx.shape[1]), np.dot(x_t,y)) 
    loss = compute_mse(y, tx, w)

    return w,loss

#LOGISTIC REGRESSION

def sigmoid(t):
    """
    Apply the sigmoid function on t.
    """
    return np.exp(t)/(1+np.exp(t))

def calculate_loss(y, tx, w):
    """
    Compute the negative log likelihood as defined in class.
    Takes as input the targeted y, the sample matrix X and the feature fector w.
    """
    return np.sum(np.log(1+np.exp(tx@w))-y*(tx@w))

def calculate_gradient(y, tx, w):
    """
    Compute the gradient of the negative log likelihood with respect to w, and the current error vector e.
    Takes as input the targeted y, the sample matrix w and the feature vector w. 
    This function is used when solving gradient based method, such that logistic_regression() and reg_logistic_regression().
    """  
    return tx.T@(sigmoid(tx@w)-y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Compute one step of gradient descent for logistic regression.
    Takes as input the targeted y, the sample matrix w, the feature w and the learning rate gamma.
    Return the feature vector w and the error defined as the negative log likelihood.
    """
    loss = calculate_loss(y,tx,w)
    grad = calculate_gradient(y,tx,w)
    w = w-gamma*grad
    return w, loss

def logistic_regression(y, tx, initial_w=None, max_iters=100, gamma=0.009, batch_size=1):
    """
    Compute an estimated solution of the problem y = sigmoid(tx @ w) and the associated error using Gradient Descent. 
    This method is equivalent to the minimization problem of finding w such that the negative log likelihood is minimal. Note that 
    this method may output a local minimum.
    Takes as input:
        * the targeted y
        * the sample matrix w
        * the initial guess for w, by default set as a vector of zeros
        * the number of iterations for Stochastic Gradient Descent
        * the learning rate gamma
        * the batch_size, which is the number of samples on which the new gradient is computed. If set to 1 it corresponds
        to Stochastic Gradient Descent, to the full number of samples it is Gradient Descent.
    """
    # init parameters
    if np.all(initial_w == None): initial_w = np.zeros(tx.shape[1])
    threshold = 1e-8
    losses = []
    y = (1 + y) / 2
    # build tx
    w = initial_w

    # start the logistic regression
    for i in range(max_iters):
        # get loss and update w.
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            w, _ = learning_by_gradient_descent(y_batch, tx_batch, w, gamma)
            # converge criterion
            losses.append(calculate_loss(y,tx,w))
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break
            #if i % int(max_iters/5) == 0:
                #print(losses[-1],i,'/{tot}'.format(tot=max_iters))

    return w,losses[-1]


    
# Regularized LOGISTIC REGRESSION

def learning_by_penalized_gradient_descent(y, tx, w, gamma, lambda_):
    """
    Compute one step of gradient descent for regularized logistic regression.
    Takes as input the targeted y, the sample matrix w, the feature w and the learning rate gamma.
    Return the feature vector w and the error defined as the negative log likelihood.
    """
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    grad = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    w = w-gamma*grad
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w=None, max_iters=100, gamma=0.009, batch_size=1):
    """
    Compute an estimated solution of the problem y = sigmoid(tx @ w) and the associated error using Gradient Descent. 
    Note that this method is a variant of logistic_regression() but with an added regularization term lambda_. 
    This method is equivalent to the minimization problem of finding w such that the negative log likelihood is minimal. Note that 
    this method may output a local minimum.
    Takes as input:
        * the targeted y
        * the sample matrix w
        * the initial guess for w, by default set as a vector of zeros
        * the number of iterations for Stochastic Gradient Descent
        * the learning rate gamma
        * the batch_size, which is the number of samples on which the new gradient is computed. If set to 1 it corresponds
        to Stochastic Gradient Descent, to the full number of samples it is Gradient Descent.
    """
    # init parameters
    if np.all(initial_w == None): initial_w = np.zeros(tx.shape[1])
    threshold = 1e-8
    losses = []
    y = (1 + y) / 2
    # build tx
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            w, loss = learning_by_penalized_gradient_descent(y_batch, tx_batch, w, gamma, lambda_)
            # converge criterion
            loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
            losses.append(loss)
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break
            #if iter % int(max_iters/5) == 0:
                #print(losses[-1],iter,'/{tot}'.format(tot=max_iters))

    return w,losses[-1]