import numpy as np
import time
import math
from numpy.linalg import qr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy.linalg.interpolative import estimate_spectral_norm

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def dbpca(X, B, k, d, profiling=False):
    """
    Online single pass pca
    Parameters
    ----------
    X : numpy array d x n 
        n is the number of data points and d is the number of attributes
    B : int
        block size or window size
    d : int
        number of attributes
    k : int
        number of principal components
    Return
    -------
    Q : a matrix, numpy array of d x k
        estimated eigen vectors
    """
    start_time = time.time()
    S0 = None
    for i in range(k):
        normal1 = np.random.normal(0, 1, d).reshape((-1,1))
        if S0 is None:
            S0 = normal1
        else:
            S0 = np.hstack((S0,normal1))
    Q, R = np.linalg.qr(S0)
    index = 0
    n = X.shape[1]
    for i in range(int(n/B)):
        # initialize matrix S of size dxk
        S = np.zeros((d, k))
        for j in range(B):
            x = X[:,index].reshape((-1,1))
            S = S + (1/B) * x.dot(x.T).dot(Q)
            index += 1
        Q, R = np.linalg.qr(S)
    if profiling:
        execution_time = time.time() - start_time ## return time in nanosecond
        return Q, execution_time
    return Q

