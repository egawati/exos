import numpy as np
import time
import math
from numpy.linalg import qr

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

def initialize_Q(d,k, mu=0, sigma = 1):
    """
    Initialize estimated eigen vectors 
    Parameters
    ----------
    d : int
        number of attributes
    k : int
        number of principle component
    Returns
    -------
    Q : a matrix, numpy array of d x k
        initial estimated eigenvectors
    """
    S0 = None
    np.random.seed(42)
    for i in range(k):
        normal1 = np.random.normal(mu, sigma, d).reshape((-1,1))
        if S0 is None:
            S0 = normal1
        else:
            S0 = np.hstack((S0,normal1))
    Q, R = np.linalg.qr(S0)
    return Q

def update_Q(W,d,k,Q):
    """
    Update estimated eigen vector for each window
    Parameters
    ----------
    W : a matrix, numpy array of d x m 
        window or buffer consisting of m data points of d attributes
    d : int
        number of attributes
    k : int
        number of principle component
    Q : a matrix, numpy array of d x k
        estimated eigenvectors from previous window
    Returns
    -------
    Q : a matrix, numpy array of d x k
        updated estimated eigenvectors
    """
    S = np.zeros((d,k))
    B = W.shape[1]
    for j in range (B):
        x = W[:,j].reshape((-1,1))
        S = S + (1/B) * x.dot(x.T).dot(Q)
    Q, R = np.linalg.qr(S)
    return Q