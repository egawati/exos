from exos.explainer.estimator import dbpca
from .common import get_outliers

import numpy as np
import time
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def concatenate_buffers(hash_d, n_streams):
    """
    Combine data points from each stream on a period of time
    Assumption: each stream produces the same number of data points
    Parameters
    ----------
    hash_d : dict
        number of attributes
    n_streams : int
        number of principle component
    Returns
    -------
    arr : np.array
        numpy array of shape n_points x total number of attributes
    -------
    """
    n_points = hash_d[0].shape[0] #the number of data points in each streams
    arr = None
    for i in range(n_points):
        new_point = hash_d[0][i]
        for j in range(1, n_streams):
            new_point = np.concatenate((new_point, hash_d[j][i]))
        if arr is None:
            arr = new_point
        else:
            arr = np.vstack((arr, new_point))
    return arr

def slice_estimating_matrix(stream_id, all_outliers, all_outliers_est, 
                            outlier_stream_idx, attributes, n_streams):
    outlier_index = outlier_stream_idx[stream_id]
    start_idx = attributes[stream_id]
    if stream_id < n_streams - 1:
        end_idx = attributes[stream_id+1]
        outliers = np.take(all_outliers[:,start_idx:end_idx], outlier_index, axis=0)
        outliers_est = np.take(all_outliers_est[:,start_idx:end_idx], outlier_index, axis=0)
    else:
        outliers = np.take(all_outliers[:,start_idx:], outlier_index, axis=0)
        outliers_est = np.take(all_outliers_est[:,start_idx:], outlier_index, axis=0)
    return outliers, outliers_est

def run_dbpca_estimator(exos_condition, est_queues, est_time_queue, buffer_queue, n_streams, Q_queue, d, k, y_queue, attributes):
    """
    Used when dbpca run in multiprocessing setting
    Parameters
    ----------
    est_queue: list of Queue
        used to return values
    buffer_queue: Queue
        used to get data (stored in a dictionary) from stream_consumer
        buffer for each stream
        Example: two streams: stream 0 (2 features) and 1 (2 features), each has 5 data points 
        {
         0: array([[ 100,   90],
                   [1000,  100],
                   [  25,   26],
                   [  27,   28],
                   [  29,   30]]), 
         1: array([[  22,   22],
                    [1000, 1000],
                    [ 225,  226],
                    [ 227,  228],
                    [ 229,  230]])
        }
    n_streams: int
        number of streams
    Q_queue: Queue
        store Q (estimated eigen vectors) 
    d: int
        total number of features (sum of number of attributes from each stream)
    k: int
    y_queue: Queue
        store information about detected outlier on current window
    attributes: tuple
        list of the start index for each stream's attributes
        example: 
        Suppose there are 3 streams: S1, S2, and S3. 
        They have 3, 2, and 3 attributes respectively.
        Then attributes = (0, 2, 4)
    """
    while True:
        exos_condition.acquire()
        try:
            print('Run estimator')
            start = time.perf_counter()
            hash_d = buffer_queue.get()
            if hash_d is None:
                return
            else:
                arr = concatenate_buffers(hash_d, n_streams)
                W = arr.T # d x m 
                Q = Q_queue.get() # d x k 
                Q = dbpca.update_Q(W,d,k,Q) # d x k 
                
                all_outliers, y_d, new_y_d = get_outliers(arr, y_queue, n_streams) ## numpy array of n_outliers x d
                Y = all_outliers.dot(Q) # m x k
                all_outliers_est = Y.dot(Q.T) # m x d
                
                Q_queue.put(Q)

                for stream_id in range(n_streams):
                    outliers, outliers_est = slice_estimating_matrix(stream_id,
                                                                     all_outliers,
                                                                     all_outliers_est,
                                                                     new_y_d,
                                                                     attributes,
                                                                     n_streams
                                                                     )
                    est_queues[stream_id].put((outliers, outliers_est, new_y_d))
                end = time.perf_counter() #end measuring estimation function
                est_time_queue.put(end - start)
                exos_condition.notify()
        finally:
            exos_condition.release()