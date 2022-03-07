import numpy as np
import pandas as pd
from skmultiflow.data import TemporalDataStream
from multiprocessing import Process, Queue, Event, Condition, Manager

from exos.explainer import temporal_neighbor
from exos.explainer.estimator import dbpca
from exos.explainer.outlying_attributes import find_outlying_attributes

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

import time

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

def run_dbpca_estimator(est_queue, hash_d, n_streams, Q_queue, d, k, y_queue, profiling=False):
    """
    Used when dbpca run in multiprocessing setting
    Parameters
    ----------
    est_queue: Queue()
        used to return values
    hash_d: dict
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
    Q_queue: Queue()
        store Q (estimated eigen vectors) 
    d: int
        total number of features (sum of number of attributes from each stream)
    k: int
    y_queue: Queue()
        store information about detected outlier on current window
    profiling: boolean
        default value: False
    """
    if profiling:
        start = time.perf_counter() #start measuring estimation function
    arr = concatenate_buffers(hash_d, n_streams)
    W = arr.T # d x m 
    Q = Q_queue.get() # d x k 
    Q = dbpca.update_Q(W,d,k,Q) # d x k 
    outliers, y_d, new_y_d = get_outliers(arr, y_queue, n_streams) ## numpy array of n_outliers x d
    Y = outliers.dot(Q) # m x k
    outliers_est = Y.dot(Q.T) # m x d
    Q_queue.put(Q)
    if profiling:
        end = time.perf_counter() #end measuring estimation function
        print(f"Running DBPCA estimation function in {end-start} second")
        est_queue.put((outliers, outliers_est, new_y_d, y_d, end-start))
    else:
        est_queue.put((outliers, outliers_est, new_y_d, y_d))

def run_temporal_neighbors(neigh_queues, buffer, stream_id, ncluster, init_data, profiling=False):
    if profiling:
        start = time.perf_counter()
    clustering = temporal_neighbor.cluster_data(buffer, ncluster, init_data)
    if profiling:
        end = time.perf_counter()
        print(f"Running temporal neighbor function in {end-start} second")
        neigh_queues[stream_id].put((stream_id, clustering, end-start))
    else:
        neigh_queues[stream_id].put((stream_id, clustering))


def run_exos1(exos_condition, buffer_queue, 
              exos1_queue, n_streams, Q_queue, d, k, y_queue, 
              n_clusters=(), n_init_data=(), profiling=False):
    """
    call EXOS here
    """
    est_queue = Queue()
    neigh_queues = [Queue()] * n_streams
    neigh_d_result = {}
    neighs = list()

    while True:
        if profiling:
            start = time.perf_counter()
        hash_d = buffer_queue.get()
        if hash_d is not None: 
            # est_start = time.perf_counter()
            # estimator = Process(target=run_dbpca_estimator, 
            #                    args=(est_queue, hash_d, n_streams, Q_queue, d , k, y_queue, profiling), 
            #                    daemon=True)
            # estimator.start()
            # print(f"It takes {time.perf_counter()-est_start} s to start estimator process")
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
            print(f"It takes {time.perf_counter()-neigh_start} s to start temporal neighbor process")
            
            njoin_start = time.perf_counter()
            # estimator.join()
            for neighbor in neighs:
                neighbor.join()
            print(f"It takes {time.perf_counter()-njoin_start} s to complete processes")

            res_start = time.perf_counter()
            # estimator_result = est_queue.get()
            for i in range(n_streams):
                if profiling:
                    stream_id, clustering, run_time = neigh_queues[i].get()
                    neigh_d_result[stream_id] = clustering, run_time
                else:
                    stream_id, clustering = neigh_queues[i].get()
                    neigh_d_result[stream_id] = clustering
            print(f"It takes {time.perf_counter()-res_start} s to get the result from queues")
            
            if profiling:
                end = time.perf_counter()
                print(f'total time is {end-start} s')
                # exos1_queue.put((estimator_result, neigh_d_result, end-start))
            # else:
            #     exos1_queue.put((estimator_result, neigh_d_result))
            # with exos_condition:
            #     exos_condition.wait()
        else:
            exos1_queue.put(hash_d)
            return 

def run_outlying_attributes(exos2_queue, stream_id, n_streams,
                            all_outliers, all_outliers_est, outlier_stream_idx,
                            inlier_centroids, attributes, feature_names, 
                            round_flag=False, multiplier=10, profiling=False):
    
    outlier_index = outlier_stream_idx[stream_id]
    start_idx = attributes[stream_id]
    
    if stream_id < n_streams - 1:
        end_idx = attributes[stream_id+1]
        outliers = np.take(all_outliers[:,start_idx:end_idx], outlier_index, axis=0)
        outliers_est = np.take(all_outliers_est[:,start_idx:end_idx], outlier_index, axis=0)
    else:
        outliers = np.take(all_outliers[:,start_idx:], outlier_index, axis=0)
        outliers_est = np.take(all_outliers_est[:,start_idx:], outlier_index, axis=0)

    d = inlier_centroids.shape[1]
    outlying_attributes = list()
    for i, outlier in enumerate(outliers):
        out_attributes = find_outlying_attributes( outlier, 
                                                   outliers_est[i],
                                                   inlier_centroids, 
                                                   d, 
                                                   feature_names, 
                                                   round_flag, 
                                                   multiplier)
        outlying_attributes.append(out_attributes)
    exos2_queue.put((stream_id, outlying_attributes))

def run_exos2(exos_condition, exos1_queue, exos2_queue, 
              result_queue, 
              n_streams, feature_names, attributes, 
              round_flag=False, multiplier=10, profiling=False):
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
        exos1 = exos1_queue.get()
        if exos1 is None:
            return
        else:
            if profiling:
                estimator_result, neigh_d_result, run_time = exos1
                outliers, outliers_est, outlier_stream_idx, ori_outlier_stream_idx, est_run_time = estimator_result
                print(f'exos 1 run time {run_time}')
            else:
                estimator_result, neigh_d_result = exos1
                outliers, outliers_est, outlier_stream_idx, ori_outlier_stream_idx = estimator_result
            explainers = list()
            for stream_id in range(n_streams):
                if profiling:
                    clustering = neigh_d_result[stream_id][0]
                else:
                    clustering = neigh_d_result[stream_id]
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
        
            results = {}
            for explainer in explainers:
                explainer.join()
                stream_id, outlying_attributes = exos2_queue.get()
                results[stream_id] = outlying_attributes
            result_queue.put(results)
            print(f'est {estimator_result}')
            print(f'temporal neighbor {neigh_d_result}')
            with exos_condition:
                exos_condition.notify_all()


if __name__ == '__main__':
    profiling = True
    round_flag = False
    multiplier = 10
    data1 = np.array([[1,2],[3,4],[5,6],[70,8], [9,10],
                      [100, 90], [1000, 100], [25, 26], [27, 28], [29, 30]])
    data2 = np.array([[111,112],[113,114],[115,116],[117,118], [1,12],
                      [22, 22], [1000, 1000], [225, 226], [227, 228], [229, 230]])
    y1 = np.array([0, 0, 0, 1, 0, 1, 1, 0, 0, 0])
    y2 = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0])

    # data1 = np.random.uniform(low=5, high=20, size=(1000,2))
    # data2 = np.random.uniform(low=100, high=200, size=(1000,2))
    # y1 = np.random.randint(2, size=1000)
    # y2 = np.random.randint(2, size=1000)

    attributes = (0, 2)
    feature_names = {0: ('A1', 'A2'), 1:('B1', 'B2')}

    ts1 = TemporalDataStream(data1, y1)
    ts2 = TemporalDataStream(data2, y2)
    #ts3 = TemporalDataStream(data3, y)
    sources = (ts1,ts2)
    n_streams = len(sources)
    d = 4 ## total attributes
    k = d
    
    if profiling:
        start = time.perf_counter()

    Q = dbpca.initialize_Q(d,k)

    print(f"Init Q {Q}")

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
                         args=(exos_condition, buffer_queue, exos1_queue, n_streams, Q_queue, d, k, y_queue, (), (), profiling), 
                         daemon=False)
    exos1.start()

    # exos2_queue = Queue()
    # result_queue = Queue()
    # exos2 = Process(target=run_exos2, 
    #                 args=(exos_condition, exos1_queue, exos2_queue, 
    #                       result_queue,
    #                       n_streams, feature_names, attributes, 
    #                       round_flag, multiplier, profiling), 
    #                 daemon=False)
    # exos2.start()

    for p in producers:
        p.join()

    for queue in queues:
        queue.put(None)
    
    consumer.join()
    buffer_queue.put(None)

    exos1.join()
    # exos2.join()
    exos1_queue.put(None)
    # result_queue.put(None)
    # while True:
    #     result = result_queue.get()
    #     if result is None:
    #         break
    #     print(f'Result {result}')

    if profiling:
        end = time.perf_counter()
        print(f'Total running time is {end-start}')

    print("Done")