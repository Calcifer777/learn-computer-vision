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

    num_samples = X.shape[0]
    num_classes = W.shape[1]
    
    for i in range(X.shape[0]):
        logits = np.matmul(X[i], W)
        logits -= np.max(logits) # numeric adjustment
        probs = np.exp(logits) / np.sum(np.exp(logits))
        loss -= np.log(probs[y[i]])
        probs[y[i]] -= 1
        dW += np.outer(X[i], probs)
        
        
    loss = loss / num_samples + reg * np.sum(np.multiply(W, W))

    dW /= num_samples
    dW += reg * 2 * W

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

    num_classes = W.shape[1]
    num_samples = X.shape[0]
    
    logits = np.matmul(X, W)
    logits -= np.max(logits, axis=1).reshape(num_samples, 1)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1).reshape(num_samples, 1)
    
    loss = np.sum( -np.log(probs[range(num_samples), y]) ) / num_samples
    loss += reg * np.sum(np.multiply(W, W))

    probs[range(num_samples), y] -= 1
    dW = np.matmul(X.T, probs) / num_samples
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
