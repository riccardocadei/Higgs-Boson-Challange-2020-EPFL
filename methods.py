import numpy as np
from orginal_helpers import *

# Gradient based methods for linear systems

def compute_mse(y, tx, w):
    e = y - tx@w
    mse = e.T.dot(e) /(2*len(e))
    return mse

def least_squares(y, tx):
    """calculate the least squares solution."""
    w = np.linalg.solve (tx.T.dot(tx),tx.T.dot(y))
    mse = compute_mse(y, tx, w)
    return mse , w 

def least_squares_gradient(y, tx, w):
    """Compute the gradient."""  
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)
    return grad, e

def least_squares_GD(y, tx, initial_w=None, max_iters=1000, gamma=0.005):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    if initial_w == None: initial_w = np.random.random(tx.shape[1])
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
        if (n_iter % 100) == 0:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters,l=loss))

    return loss, w

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

def least_squares_SGD(y, tx, initial_w=None, batch_size=1, max_iters=1000, gamma=0.005):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    if initial_w == None: initial_w = np.random.random(tx.shape[1])
    ws = [initial_w]
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
            ws.append(w)
            losses.append(loss)

        if n_iter % 100 == 0:
            print("SGD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
    return loss, ws

# RIDGE REGRESSION

def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations
    """
    x_t = tx.T
    lambd = lambda_ * 2 * len(y)
    w = np.dot(np.dot(np.linalg.inv(np.dot(x_t, tx) + lambd * np.eye(tx.shape[1])), x_t), y)
    loss = compute_mse(y, tx, w)
    return loss,w




#LOGISTIC REGRESSION

def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t)/(1+np.exp(t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    return np.sum(np.log(1+np.exp(tx@w))-y*(tx@w))/len(y)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T@(sigmoid(tx@w)-y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y,tx,w)
    grad = calculate_gradient(y,tx,w)
    w = w-gamma*grad
    return loss, w

def logistic_regression(y, tx, initial_w=None, batch_size=1, max_iter=100, gamma=0.009):
    # init parameters
    if initial_w == None: initial_w = np.random.random(tx.shape[1])
    threshold = 1e-8
    losses = []
    y[y==-1]=0
    # build tx
    w = initial_w

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            _, w = learning_by_gradient_descent(y_batch, tx_batch, w, gamma)
            # converge criterion
            losses.append(calculate_loss(y,tx,w))
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break
            if iter % int(max_iter/10) == 0:
                print(losses[-1],iter,'/{tot}'.format(tot=max_iter))

    return losses[-1], w
    
def learning_by_stochastic_gradient_descent(y, tx, w, gamma,minibatch_y,minibatch_tx):
    """
    Do one step of stochastic gradient descenr using logistic regression.
    Return the loss and the updated w.
    """
    loss = compute_loss(y, tx, w)
    grad = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
    w = w - gamma * grad
    return loss, w






    
# Regularized LOGISTIC REGRESSION

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient."""
    num_samples = y.shape[0]
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    print(w.shape, gradient.shape)
    w -= gamma * gradient
    return loss, w

def logistic_regression_penalized_gradient_descent_demo(y, x, initial_w=None):
    # init parameters
    if initial_w == None: initial_w = np.random.random(tx.shape[1])
    max_iter = 500
    gamma = 0.01
    lambda_ = 0.1
    threshold = 1e-6
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    return loss, w
    
