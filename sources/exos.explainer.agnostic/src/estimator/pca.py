import numpy as np
import pandas as pd 
import time
import math
from sklearn.metrics import mean_squared_error
from scipy.linalg.interpolative import estimate_spectral_norm

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

def compute_principal_components(cov, C, variance_explained=1, profiling=False):
    eig_values, eig_vectors = eig(cov)
    k = len(eig_values)
    if variance_explained != 1:
        trace = sum(eig_values)
        var_explained = 0
        for i, val in enumerate(eig_values):
            var_explained += val/trace
            if var_explained >= variance_explained:
                k = i
    return eig_vectors, k


def project_data(C, eig_vectors, k, profiling=False):
    """
    C is a matrix n x d, where n is number of data points, d is number of attributes
    eig_vectors of d x k
    """
    # now project X 
    Y = eig_vectors[:,0:k].T.dot(C.T)
    return Y


def do_pca(C, variance_explained=1, profiling=False):
    """
    C is a matrix n x d, where n is number of data points, d is number of attributes
    """
    cov = np.cov(C.T)
    eig_vectors, k = compute_principal_components(cov, C, variance_explained, profiling)
    Y = project_data(C, eig_vectors, k, profiling)
    return Y, eig_vectors, k


def reconstruct_X(Y, eig_vectors, k):
    return eig_vectors[:,0:k].dot(Y)


def naive_pca_per_window(B, X, k, d, profiling=False):
    """
    Run PCA on every window
    Parameters
    ----------
    B : int
        block size or window size
    X : numpy array n x d
        n is the number of data points and d is the number of attributes
    k : int
        number of principal components
    d : int
    	number of attributes (features)
    Outputs
    --------
    eig_vectors: list
    	list of eigen vectors
    Ys: list
    	list of principal components
    Ms: list
    	list of mean for each window
    """
    start_time = time.time()
    eig_vectors = list()
    Ys = list()
    Ms = list()
    n = X.shape[0]
    for i in range(int(n/B)):
        start = i*B
        end = (i*B)+B
        W = X[start:end,:]
        M = np.mean(W, axis=0) # first pass on X
        C = W - M              # second pass on X
        Y, eig_vector, k = do_pca(C)
        Ys.append(Y)
        eig_vectors.append(eig_vector)
        Ms.append(M)
        
    if profiling:
        execution_time = time.time() - start_time ## return time in nanosecond
        return eig_vectors, Ys, Ms, execution_time
        
    return eig_vectors, Ys, Ms


def naive_pca_per_window_with_f(B, X, k, d, f, profiling=False):
    """
    Run PCA on every window with forgetting factor
    Parameters
    ----------
    B : int
        block size or window size
    X : numpy array n x d
        n is the number of data points and d is the number of attributes
    k : int
        number of principal components
    d : int
    	number of attributes (features)
    f : float
        forgetting factor (0,1], 1 basically means no forgetting factor
    
    Outputs
    --------
    eig_vectors: list
    	list of eigen vectors
    Ys: list
    	list of principal components
    Ms: list
    	list of mean for each window
    """
    start_time = time.time()
    eig_vectors = list()
    Ys = list()
    Ms = list()
    Ws = list()
    n = X.shape[0]
    t0 = B
    for i in range(int(n/B)):
        start = i*B
        end = (i*B)+B
        delta_t = t0-i ## delta_t represent the age of the point
        W = X[start:end,:] * (math.pow(2, -f*delta_t))
        M = np.mean(W, axis=0) # first pass on X
        C = W - M              # second pass on X
        Y, eig_vector, k = do_pca(C)
        Ws.append(W)
        Ys.append(Y)
        eig_vectors.append(eig_vector)
        Ms.append(M)
        
    if profiling:
        execution_time = time.time() - start_time ## return time in nanosecond
        return eig_vectors, Ys, Ms, Ws, execution_time
        
    return eig_vectors, Ys, Ms, Ws