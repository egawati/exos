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