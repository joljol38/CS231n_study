from builtins import range
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    

    for i in range(X.shape[0]):
      score = X[i].dot(W)
      score = score - max(score) # not run into numeric instability!!!
      loss_i = -score[y[i]] + np.log(sum(np.exp(score)))
      loss += loss_i

      for j in range(W.shape[1]):
        softmax = np.exp(score[j]) / sum(np.exp(score))
        dW[:,j] += softmax * X[i]
      dW[:,y[i]] -= X[i]

      loss /= X.shape[0]
      dW /= X.shape[0]

      loss += 0.5 * reg * np.sum(W * W)
      dW += 2 * reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    score = X.dot(W)
    score -= np.max(score)

    score_exp = np.exp(score)
    score_exp_sum = np.sum(score_exp, axis = 1)
    
    loss = score_exp[range(X.shape[0]), y] / score_exp_sum
    loss = -np.sum(np.log(loss)) / X.shape[0] + (reg * np.sum(W * W))

    dS = score_exp / score_exp_sum.reshape(-1, 1)
    dS[range(X.shape[0]), y] -= 1

    dW = (X.T).dot(dS)

    loss /= X.shape[0]
    dW /= X.shape[0]

    loss += 0.5 * reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
