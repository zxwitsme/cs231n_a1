import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  dW1 = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_class = W.shape[1]
  num_train, dim = X.shape
  s = X.dot(W)
  s_max = np.reshape(np.max(s,axis=1),(num_train,1))
  p = np.exp(s-s_max)/np.sum(np.exp(s-s_max),axis=1,keepdims=True)
  y_true = np.zeros_like(p)
  y_true[np.arange(num_train), y] = 1.0
  for i in xrange(num_train):
    for j in xrange(num_class):
      loss += -(y_true[i,j]*np.log(p[i,j]))
      dW1[:, j] = -(y_true[i,j]-p[i,j])*X[i,:]
    dW += dW1
  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)
  dW /= num_train
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train, dim = X.shape

  s = X.dot(W)
  s = X.dot(W)
  s_max = np.reshape(np.max(s,axis=1),(num_train,1))
  p = np.exp(s-s_max)/np.sum(np.exp(s-s_max),axis=1,keepdims=True)
  y_true = np.zeros_like(p)
  y_true[np.arange(num_train), y] = 1.0
  loss += -np.sum(y_true*np.log(p))/num_train+0.5*reg*np.sum(W*W)
  dW += -np.dot(X.T, y_true-p)/num_train+reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

