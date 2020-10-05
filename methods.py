import numpy as np

# Gradient based methods for linear systems

def compute_mse(y_tr,tx_tr,weight):
    return np.linalg.norm(y_tr-tx_tr.dot(weight))/len(y_tr)

def least_squares(y, tx):
    """calculate the least squares solution."""
    w = np.linalg.solve (tx.T.dot(tx),tx.T.dot(y))
    mse = compute_mse(y, tx, w)
    return mse , w 

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)
    return grad, e

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws


#LOGISTIC REGRESSION

def sigmoid(t):
    return np.exp(t)/(1+np.exp(t))

def calculate_loss(y, tx, w):
  pred = sigmoid(tx.dot(w))
  loss = -y.T.dot(np.log(pred))+(1-y).T.dot(np.log(1-pred))
  return np.squeeze(-loss)

def calculate_gradient(y, tx, w):
    return tx.T.dot(sigmoid(tx.dot(w))-y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w = w - gamma*grad
    return loss, w

def logistic_regression_gradient_descent_demo(y, x, max_iter = 10000,threshold = 1e-8,gamma = 0.01):
    # init parameters
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_gradient_descent")
    print("loss={l}".format(l=calculate_loss(y, tx, w)))
    
def learning_by_stochastic_gradient_descent(y, tx, w, gamma,minibatch_y,minibatch_tx):
    """
    Do one step of stochastic gradient descenr using logistic regression.
    Return the loss and the updated w.
    """
    loss = compute_loss(y, tx, w)
    grad = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
    w = w - gamma * grad
    return loss, w

def logistic_regression_gradient_descent_demo(y, x, batch_size, max_iter = 10000,threshold = 1e-8,gamma = 0.01):
    # init parameters
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
          loss, w = learning_by_stochastic_gradient_descent(y, tx, w, gamma)
          # log info
          if iter % 100 == 0:
              print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
          # converge criterion
          losses.append(loss)
          if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
              break
    # visualization
    visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_gradient_descent")
    print("loss={l}".format(l=calculate_loss(y, tx, w)))
    
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
    w -= gamma * gradient
    return loss, w

def logistic_regression_penalized_gradient_descent_demo(y, x):
    # init parameters
    max_iter = 10000
    gamma = 0.01
    lambda_ = 0.1
    threshold = 1e-8
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
    visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_penalized_gradient_descent")
    print("loss={l}".format(l=calculate_loss(y, tx, w)))
    