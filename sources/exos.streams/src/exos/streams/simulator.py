import numpy as np
import pandas as pd
from skmultiflow.data import TemporalDataStream
from multiprocessing import Process, Queue, Condition

from exos.explainer.estimator import dbpca

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

def run_dbpca_estimator(est_queue, hash_d, n_streams, Q_queue, d, k):
    arr = concatenate_buffers(hash_d, n_streams)
    W = arr.T
    Q = Q_queue.get()
    Q = dbpca.update_Q(W,d,k,Q)
    Q_queue.put(Q)
    est_queue.put(Q)

def run_temporal_neighbors(neigh_queues, buffer, stream_id, ncluster, init_data):
    clustering = temporal_neighbor.cluster_data(buffer, ncluster, init_data)
    neigh_queues[stream_id].put((stream_id, clustering))

def run_experiment(buffer_queue, experiment_queue, n_streams, Q_queue, d, k, n_clusters=(), n_init_data=()):
    """
    call EXOS here
    """
    est_queue = Queue()
    neigh_queues = [Queue()] * n_streams
    neigh_d_result = {}
    neighs = list()
    
    while True:
        hash_d = buffer_queue.get()
        if hash_d is not None:
            estimator = Process(target=run_dbpca_estimator, 
                                args=(est_queue, hash_d, n_streams, Q_queue, d , k), 
                                daemon=True)
            estimator.start()
            ## get the value put into est_queue in run_estimator
            
            for i in range(n_streams):
                n_cluster = 2
                if n_clusters:
                    n_cluster = n_clusters[i]
                
                init_data = None 
                if n_init_data:
                    init_data = n_init_data[i]

                neighbor = Process(target=run_temporal_neighbors, 
                                   args=(neigh_queues, hash_d[i], i, n_cluster, init_data), 
                                   daemon=True)
                neighbor.start()
                neighs.append(neighbor)
            
            estimator.join()
            for neighbor in neighs:
                neighbor.join()

            estimator_result = est_queue.get()
            for i in range(n_streams):
                stream_id, clustering = neigh_queues[i].get()
                neigh_d_result[stream_id] = clustering
            experiment_queue.put((estimator_result, neigh_d_result))

        else:
            experiment_queue.put(hash_d)
            return 