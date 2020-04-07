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
#         c = np.matmul(X[i],W)
#         c -= np.amax(c)
#         e_c = np.exp(c)
#         denom = np.sum(e_c)
#         #Nice fact: we know that the largest element in c will also be the largest softmax value, so we only
#         # need to transform that one value. 
#         sm_c = e_c/denom
# 
#         loss1 += -np.log(sm_c[y[i]])

        # Need to make this whole dang thing more numerically stable. 
        c = np.matmul(X[i],W)
        c -= np.amax(c)
        e_c = np.exp(c)
        denom = np.sum(e_c)
        sm_c = e_c/denom

        loss += np.log(denom) - c[y[i]]
#        print(-np.log(sm_c[y[i]]) - (np.log(denom)-c[y[i]]))

        """They are basically the same value"""

        # now computing some gradients
        dL_ds = sm_c
        dL_ds[y[i]] -= 1
        #note that sm_c is modified now!
        """ #ah, something fundamentally different is happening with numpy. When an array element
        is changed, it's really changed for good. And it changes for all pointers pointing to same object.
        yikes. Actually it's the same with python lists. Anything pointing to And underlying object can
        change that underlying object for all things that point to it. Alas."""
#        import pdb; pdb.set_trace()
        """Okay I just coudln't bear the for loops..."""
        dW_update = np.matmul(X[i].reshape(1,X.shape[1]).T,dL_ds[np.newaxis,:])
        dW+=dW_update
        #         for n in range(W.shape[0]):
#             for m in range(W.shape[1]):
#                 if m == y[i]:
#                     dW[n,m] += X[i,n]*(sm_c[m]-e_c[m])
#                 else:
#                     dW[n,m] += X[i,n]*sm_c[m]

        # should be numerically unstable I think.

    loss /= X.shape[0]
    loss += reg*np.sum(W*W)

    dW /= X.shape[0]
    dW += reg*2*W
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
    c = np.matmul(X,W)
    c -= np.amax(c,axis = -1)[:,np.newaxis]
    e_c = np.exp(c)
    denom = np.sum(e_c,axis = -1)[:,np.newaxis]
    sm_c = e_c/denom

    
    loss_vec = np.log(denom) - c[range(c.shape[0]),y][:,np.newaxis]

    # now computing some gradients
    dL_ds = sm_c
    dL_ds[range(dL_ds.shape[0]),y] -= 1
    dW_update = np.matmul(X.T,dL_ds)
    dW+=dW_update
    
    loss = np.sum(loss_vec)
    loss /= X.shape[0]

    loss += reg*np.sum(W*W)
    print(dW)
    dW /= X.shape[0]

    dW += reg*2*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    print(dW)
    return loss, dW
