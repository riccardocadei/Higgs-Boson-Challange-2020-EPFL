import numpy as np

# Gradient based methods for linear systems

def compute_mse(y, tx, w):
    e = y - tx@w
    mse = e.T.dot(e) /(2*len(e))
    return mse

def least_squares(y, tx):
    """calculate the least squares solution."""
    w = np.linalg.solve (tx.T.dot(tx),tx.T.dot(y))
    mse = compute_mse(y, tx, w)
    return w, mse 

def least_squares_gradient(y, tx, w):
    """Compute the gradient."""  
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)
    return grad, e

def least_squares_GD(y, tx, initial_w=None, max_iters=50, gamma=0.005,plot=False):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    if np.all(initial_w == None): initial_w = np.random.random(tx.shape[1])    
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
        #if (n_iter % 100) == 0:
            #print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters,l=loss))
    if plot:
        return w, losses
    else:
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

def least_squares_SGD(y, tx, initial_w=None, batch_size=1, max_iters=50, gamma=0.00005,plot=False):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    if np.all(initial_w == None): initial_w = np.random.random(tx.shape[1])    
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

        if n_iter % 10 == 0:
            print("SGD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
    if plot:
        return w, losses
    else:
        return w,loss
# RIDGE REGRESSION

def ridge_regression(y, tx, lambda_,plot=False):
    """ Ridge regression using normal equations
    """
    x_t = tx.T
    lambd = lambda_ * 2 * len(y)
    w = np.linalg.solve (np.dot(x_t, tx) + lambd * np.eye(tx.shape[1]), np.dot(x_t,y)) 
    loss = compute_mse(y, tx, w)
    if plot:
        return w, losses
    else:
        return w,loss

#LOGISTIC REGRESSION

def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t)/(1+np.exp(t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    return np.sum(np.log(1+np.exp(tx@w))-y*(tx@w))

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T@(sigmoid(tx@w)-y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y,tx,w)
    grad = calculate_gradient(y,tx,w)
    w = w-gamma*grad
    return w, loss

def logistic_regression(y, tx, initial_w=None, max_iters=100, gamma=0.009, batch_size=1,plot=False):
    # init parameters
    if np.all(initial_w == None): initial_w = np.random.random(tx.shape[1])
    threshold = 1e-8
    losses = []
    y[y==-1]=0
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
            if i % int(max_iters/5) == 0:
                print(losses[-1],i,'/{tot}'.format(tot=max_iters))

    if plot:
        return w, losses
    else:
        return w,losses[-1]


    
# Regularized LOGISTIC REGRESSION

def learning_by_penalized_gradient_descent(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    grad = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    w = w-gamma*grad
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w=None, max_iters=100, gamma=0.009, batch_size=1,plot=False):
    # init parameters
    if np.all(initial_w == None): initial_w = np.random.random(tx.shape[1])
    threshold = 1e-8
    losses = []
    y[y==-1]=0
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
            if iter % int(max_iters/100) == 0:
                print(losses[-1],iter,'/{tot}'.format(tot=max_iters))

    if plot:
        return w, losses
    else:
        return w,losses[-1]