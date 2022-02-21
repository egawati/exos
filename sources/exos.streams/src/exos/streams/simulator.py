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

def get_outliers(arr, y_queue, n_streams):
    y_d = y_queue.get()
    new_y_d = {}
    new_y_d[0] = list()
    outlier_index = set(y_d[0])    
    
    for i in range(1, n_streams):
        outlier_index.update(y_d[i])
        new_y_d[i] = list()
    outlier_index = list(outlier_index)
    
    for i in range(n_streams):
        for idx in y_d[i]:
            if idx in outlier_index:
                new_y_d[i].append(outlier_index.index(idx))
    
    outliers= np.take(arr, outlier_index, axis=0)
    return outliers, y_d, new_y_d 

def run_dbpca_estimator(est_queue, hash_d, n_streams, Q_queue, d, k, y_queue):
    arr = concatenate_buffers(hash_d, n_streams)
    W = arr.T # d x m 
    Q = Q_queue.get() # d x k 
    Q = dbpca.update_Q(W,d,k,Q) # d x k 
    outliers, y_d, new_y_d = get_outliers(arr, y_queue, n_streams) ## numpy array of n_outliers x d
    Y = outliers.dot(Q) # m x k
    outliers_est = Y.dot(Q.T) # m x d
    Q_queue.put(Q)
    est_queue.put((outliers_est, new_y_d, y_d))

def run_temporal_neighbors(neigh_queues, buffer, stream_id, ncluster, init_data):
    clustering = temporal_neighbor.cluster_data(buffer, ncluster, init_data)
    neigh_queues[stream_id].put((stream_id, clustering))


def run_exos1(exos_condition, buffer_queue, 
              exos1_queue, n_streams, Q_queue, d, k, y_queue, 
              n_clusters=(), n_init_data=()):
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
                                args=(est_queue, hash_d, n_streams, Q_queue, d , k, y_queue), 
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
            exos1_queue.put((estimator_result, neigh_d_result))
            with exos_condition:
                exos_condition.wait()
        else:
            exos1_queue.put(hash_d)
            return 

def run_exos2(exos_condition, exos1_queue, exos2_queue, n_streams, attributes):
    while True:
        exos1 = exos1_queue.get()
        if exos1 is None:
            return
        else:
            print(f"{exos1}")
            with exos_condition:
                exos_condition.notify_all()


if __name__ == '__main__':
    data1 = np.array([[1,2],[3,4],[5,6],[7,8], [9,10],
                      [21, 22], [23, 24], [25, 26], [27, 28], [29, 30]])
    data2 = np.array([[111,112],[113,114],[115,116],[117,118], [119,120],
                      [221, 222], [223, 224], [225, 226], [227, 228], [229, 230]])

    y1 = np.array([0, 0, 0, 1, 0, 1, 1, 0, 0, 0])
    y2 = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    ts1 = TemporalDataStream(data1, y1)
    ts2 = TemporalDataStream(data2, y2)
    attributes = {0: ('A1', 'A2'), 1:('B1', 'B2')}
    #ts3 = TemporalDataStream(data3, y)

    sources = (ts1, ts2)
    n_streams = len(sources)
    d = 4 ## total attributes
    k = d
    Q = dbpca.initialize_Q(d,k)

    print(f"Init Q {Q}")

    ### Start the multiprocessing 
    queues = [Queue()] * n_streams
    condition = Condition()

    producers = [Process(target=stream_producer, args=(condition, queues, sources[i], i, 5), daemon=True) for i in range(n_streams)]
    for p in producers:
        p.start()
    
    buffer_queue = Queue() 
    y_queue = Queue()        
    consumer = Process(target=stream_consumer, args=(condition, queues, buffer_queue, y_queue), daemon=True)
    consumer.start()

    exos_condition = Condition()
    exos1_queue = Queue()
    Q_queue = Queue()
    Q_queue.put(Q)
    exos1 = Process(target=run_exos1, 
                         args=(exos_condition, buffer_queue, exos1_queue, n_streams, Q_queue, d, k, y_queue), 
                         daemon=False)
    exos1.start()

    exos2_queue = Queue()
    exos2 = Process(target=run_exos2, 
                         args=(exos_condition, exos1_queue, exos2_queue, n_streams, attributes), 
                         daemon=True)
    exos2.start()

    for p in producers:
        p.join()

    for queue in queues:
        queue.put(None)
    
    consumer.join()
    buffer_queue.put(None)

    exos1.join()
    exos2.join()
    exos1_queue.put(None)
    exos1_queue.put(None)

    print("Done")