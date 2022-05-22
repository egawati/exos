import numpy as np


def get_outlier_indices(y_d, n_streams):
    new_y_d = {}
    new_y_d[0] = list()

    outlier_indices = {}
    outlier_indices[0] = [int(val) for val in y_d[0]]

    outlier_index = set(y_d[0])    
    
    for i in range(1, n_streams):
        outlier_index.update(y_d[i])
        new_y_d[i] = list()
        outlier_indices[i] = [int(val) for val in y_d[i]]

    outlier_index = list(outlier_index)
    
    for i in range(n_streams):
        for idx in y_d[i]:
            if idx in outlier_index:
                new_y_d[i].append(outlier_index.index(idx))

    return new_y_d, outlier_indices, outlier_index


def get_outliers(arr, outlier_index):
    """
    arr : numpy arry
        a matrik of N x D
    y_d : dictionary
        key-value pair, key refers to stream_id, value refers to list of outlier index
        y_d{0: array([2, 3]), 1: array([3])}
    n_streams: int 
        number of streams
    """
    outliers= np.take(arr, outlier_index, axis=0)
    return outliers