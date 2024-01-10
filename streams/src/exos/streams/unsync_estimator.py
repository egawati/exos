from exos.explainer.estimator import dbpca

import os
import numpy as np
import time
import sys
import setproctitle

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

import math

def find_lcm_of_numbers(numbers):
    lcm = 1
    for number in numbers:
        lcm = lcm * number // math.gcd(lcm, number)
    return lcm

def get_idx_stream_indicators(arrival_rates):
    shared_delta_time = find_lcm_of_numbers(arrival_rates)
    idx_stream_indicators = [shared_delta_time//arrival_rate for arrival_rate in arrival_rates]
    return idx_stream_indicators

def concatenate_buffers(hash_d, idx_stream_indicators):
    """
    Combine data points from each stream on a period of time
    Assumption: each stream produces the same number of data points
    Parameters
    ----------
    hash_d : dict
        Example: two streams: stream 0 (2 features) and 1 (2 features), each has 5 and 3 data points respectively
        {
         0: array([[ 100,   90],
                   [1000,  100],
                   [  25,   26],
                   [  27,   28],
                   [  29,   30]]), 
         1: array([[  22,   22],
                    [ 227,  228],
                    [ 229,  230]])
        }
    Returns
    -------
    concat_arr : np.array
        numpy array of shape n_points x total number of attributes
    -------
    """ 
    new_hash_d = {}
    stream_ids = list(hash_d.keys())
    stream_ids.sort()
    for stream_id in stream_ids:
        a = idx_stream_indicators[stream_id]
        arr = hash_d[stream_id]
        n = arr.shape[0] + 1
        new_hash_d[stream_id] = arr[(np.arange(1, n) % a == 0)]
    concat_arr = np.hstack(list(new_hash_d.values()))
    return concat_arr

def get_outlier_indices(y_d, n_streams):
    """
    y_d : dictionary
        key-value pair, key refers to stream_id, value refers to list of outlier index
        y_d{0: array([2, 3]), 1: array([3])}
    n_streams: int 
        number of streams
    """
    outlier_indices = {}
    outlier_indices[0] = [int(val) for val in y_d[0]]

    outlier_index = set(y_d[0])    
    
    for i in range(1, n_streams):
        outlier_index.update(y_d[i])
        outlier_indices[i] = [int(val) for val in y_d[i]]
    outlier_index = list(outlier_index)

    return outlier_indices, outlier_index


def get_outliers(hash_d, outlier_indices):
    outliers_dict = {}
    for stream_id, indices in outlier_indices.items():
        stream_arr = hash_d[stream_id]
        if indices:
            outliers_values = stream_arr[indices].copy()
            outliers_dict[stream_id] = outliers_values
        else:
            outliers_dict[stream_id] = np.array([])
    return outliers_dict

def update_Q(Q, arr, d, k):
    """
    Q : d x k
    arr: N x d
    """
    W = arr.T.copy() # d x N             
    Q = dbpca.update_Q(W,d,k,Q) # d x k 
    return Q

def slice_Q_per_stream(Q, attributes, n_streams):
    """
    Q : d x k
    """
    sliced_Q = {}
    for stream_id in range(n_streams):
        start_idx = attributes[stream_id]
        if stream_id < n_streams - 1:
            end_idx = attributes[stream_id + 1]
            sliced_Q[stream_id] = Q[start_idx:end_idx, :]
        else:
            sliced_Q[stream_id] = Q[start_idx:, :]
    return sliced_Q


def compute_outliers_est(Q_sliced, outliers):
    """
    Q_slides = d' x k
    outliers = n x d
    """
    Y = outliers.dot(Q_sliced) # m x k
    outliers_est = Y.dot(Q_sliced.T) # m x d
    return outliers_est

def run_dbpca_unsync_estimator(value, neigh_condition, exos_condition, 
                        est_queues, est_time_queue, buffer_queue, 
                        n_streams, Q_queue, d, k, y_queue, 
                        attributes, arrival_rates):
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
    pid = os.getpid()
    setproctitle.setproctitle("Exos.Estimator")
    while True:
        start = time.perf_counter()
        try:
            hash_d = buffer_queue.get()
            y_d = y_queue.get()
            
            if hash_d is None:
                for stream_id in range(n_streams):
                    est_queues[stream_id].put(None)
                est_time_queue.put(None)
                Q_queue.put(None)
                logging.info(f"estimator done\n")
                break
            else:
                logging.info('Run estimator\n')
                idx_stream_indicators = get_idx_stream_indicators(arrival_rates)
                arr = concatenate_buffers(hash_d, idx_stream_indicators)
                
                outlier_indices, outlier_index = get_outlier_indices(y_d, n_streams)

                Q = Q_queue.get()
                Q = update_Q(Q, arr, d, k)
                sliced_Q = slice_Q_per_stream(Q, attributes, n_streams)

                Q_queue.put(Q)

                outliers_map = get_outliers(hash_d, outlier_indices)
                
                for stream_id, outliers in outliers_map.items():
                    if outliers.shape[0] > 0:
                        outliers_est = compute_outliers_est(sliced_Q[stream_id], outliers_map[stream_id])
                        est_queues[stream_id].put((outliers, outliers_est, outlier_indices))
                    else:
                        est_queues[stream_id].put((outliers, outliers, outlier_indices))
                
                end = time.perf_counter() #end measuring estimation function
                est_time_queue.put(end - start)
                with exos_condition:
                    exos_condition.wait_for(lambda : value.value==0)
                with value.get_lock():
                    value.value = n_streams
                logging.info("Ready to waking up temporal neighbor\n")
                with neigh_condition:
                    neigh_condition.notify_all()
                logging.info("estimator --> temporal neighbor woken\n")
        except Exception as e:
            logging.error(f'Exception at estimator {e}')
    logging.info(f'estimator {pid} exit\n')
    value.value = -1 ### need to set value.value == -1 to make sure things work 
    sys.stdout.flush()