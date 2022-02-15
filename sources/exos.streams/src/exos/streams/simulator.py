import numpy as np
import pandas as pd
from skmultiflow.data import TemporalDataStream
from multiprocessing import Process, Queue, Condition

from .generator import multiple_csv_to_streams

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
    k : int
        number of principle component
    Returns
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


def run_experiment(buffer_queue, experiment_queue, n_streams):
    while True:
        d = buffer_queue.get()
        if d is not None:
            arr = concatenate_buffers(d, n_streams)
            experiment_queue.put(arr)
        else:
            experiment_queue.put(d)
            return            

def run_dbpca_experiment(buffer_queue, experiment_queue, n_streams, window_size, Q):
    """
    call EXOS with dpbca here
    """
    B = window_size

    while True:
        d = buffer_queue.get()
        if d is not None:
            arr = concatenate_buffers(d, n_streams)
            W = arr.T
            ## TODO : call exos here
            experiment_queue.put(arr)
        else:
            experiment_queue.put(d)
            return 