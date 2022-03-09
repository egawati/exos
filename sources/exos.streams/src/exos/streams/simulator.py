import numpy as np
import pandas as pd
from skmultiflow.data import TemporalDataStream
from multiprocessing import Process, Queue, Condition

from .generator import stream_producer, stream_consumer
from .estimator import run_dbpca_estimator
from .temporal_neighbor import run_temporal_neighbors
from .outlying_attributes import run_outlying_attributes

import time

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
        start = time.perf_counter()
        hash_d = buffer_queue.get()
        if hash_d is not None: 
            est_start = time.perf_counter()
            estimator = Process(target=run_dbpca_estimator, 
                               args=(est_queue, hash_d, n_streams, Q_queue, d , k, y_queue, profiling), 
                               daemon=True)
            estimator.start()
            logging.info(f"It takes {time.perf_counter()-est_start} s to start estimator process")
            neigh_start = time.perf_counter()
            n_cluster = 2
            init_data = None 
            for i in range(n_streams):
                if n_clusters:
                    n_cluster = n_clusters[i]
                
                if n_init_data:
                    init_data = n_init_data[i]

                neighbor = Process(target=run_temporal_neighbors, 
                                   args=(neigh_queues, hash_d[i], i, n_cluster, init_data, profiling), 
                                   daemon=True)
                neighbor.start()
                neighs.append(neighbor)
            logging.info(f"It takes {time.perf_counter()-neigh_start} s to start temporal neighbor process")
            
            njoin_start = time.perf_counter()
            estimator.join()
            for neighbor in neighs:
                neighbor.join()
            logging.info(f"It takes {time.perf_counter()-njoin_start} s to complete processes")

            res_start = time.perf_counter()
            estimator_result = est_queue.get()
            for i in range(n_streams):
                stream_id, clustering, run_time = neigh_queues[i].get()
                neigh_d_result[stream_id] = clustering, run_time
            logging.info(f"It takes {time.perf_counter()-res_start} s to get the result from queues")
            
            end = time.perf_counter()
            logging.info(f'total time is {end-start} s')
            exos1_queue.put((estimator_result, neigh_d_result, end-start))

            with exos_condition:
                exos_condition.wait()
        else:
            exos1_queue.put(None)
            return 



def run_exos2(exos_condition, exos1_queue, exos2_queue, 
              result_queue, 
              n_streams, feature_names, attributes, 
              round_flag=False, multiplier=10):
    """
    if profiling:
        estimator_result(
                            array([[  7,   8, 117, 118],
                                   [  9,  10, 119, 120]])
                            array([[  7.,   8., 117., 118.], 
                                   [  9.,  10., 119., 120.]]), 
                            {0: [0], 1: [1]},
                            {1: array([4]), 0: array([3])},
                            0.0010070980001728458
                        )
        neigh_d_result {
                        1: (<exos.explainer.temporal_neighbor.SequentialKMeans object at 0x127537730>, 0.00039398099988829927), 
                        0: (<exos.explainer.temporal_neighbor.SequentialKMeans object at 0x127537c10>, 0.0005255490000308782)
                        }
    """
    while True:
        start = time.perf_counter()
        exos1 = exos1_queue.get()
        if exos1 is None:
            return
        else:
            estimator_result, neigh_d_result, exos1_run_time = exos1
            outliers, outliers_est, outlier_stream_idx, ori_outlier_stream_idx, est_run_time = estimator_result

            results = {}
            if outliers.size > 0
                explainers = list()
                for stream_id in range(n_streams):
                    clustering = neigh_d_result[stream_id][0]
                    inlier_centroids = [cluster.centroid for cluster in clustering.clusters]
                    inlier_centroids = np.array(inlier_centroids)
                    
                    explainer = Process(target = run_outlying_attributes, 
                                        args = (exos2_queue, stream_id, n_streams,
                                                outliers, outliers_est, outlier_stream_idx,
                                                inlier_centroids, attributes, feature_names, 
                                                round_flag, multiplier, profiling),
                                        daemon = True)
                    explainer.start()
                    explainers.append(explainer)
            
                
                for explainer in explainers:
                    explainer.join()
                    stream_id, outlying_attributes = exos2_queue.get()
                    results[stream_id] = outlying_attributes
            
            end = time.perf_counter()
            result_queue.put((results, exos1_run_time, end-start))
            with exos_condition:
                exos_condition.notify_all()

def run_exos_simulator(sources, d, k, attributes, feature_names, 
                       n_clusters = (), n_init_data = (),
                       multiplier = 10, profiling=True, round_flag=True):
    """
    sources : list
        list of the TemporalDataStream objects
    d : int
        number of attributes
    k : int
        number of principle components to used
    attributes: tuple
        list of the start index for each stream's attributes
        example: 
        Suppose there are 3 streams: S1, S2, and S3. 
        They have 3, 2, and 3 attributes respectively.
        Then attributes = (0, 2, 4)
    feature_names : dictionary
        key = stream id
        value = list of attribute names
    n_clusters : tuple
        list of the number of clusters in each stream
    n_init_data = 
    multiplier: int
        the number of data points to sample when creating inlier/outlier class
    """
    start = time.perf_counter()

    Q = dbpca.initialize_Q(d,k)

    ### Start the multiprocessing 
    queues = [Queue()] * n_streams
    condition = Condition()

    producers = [Process(target=stream_producer, args=(condition, queues, sources[i], i, 500), daemon=True) for i in range(n_streams)]
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
                    args=(exos_condition, buffer_queue, exos1_queue, 
                          n_streams, Q_queue, d, k, y_queue, n_clusters, 
                          n_clusters, n_init_data), 
                    daemon=False)
    exos1.start()

    exos2_queue = Queue()
    result_queue = Queue()
    exos2 = Process(target=run_exos2, 
                    args=(exos_condition, exos1_queue, exos2_queue, 
                          result_queue,
                          n_streams, feature_names, attributes, 
                          round_flag, multiplier), 
                    daemon=False)
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
    

    results = list()
    result_queue.put(None)
    while True:
        result = result_queue.get()
        if result is None:
            break
        else:
            results.append(result)

    end = time.perf_counter()
    logging.info(f'Total running time is {end-start}')
    logging.info("Done")
    return results