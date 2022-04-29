import numpy as np
import pandas as pd
from skmultiflow.data import TemporalDataStream


from multiprocessing import Process, Condition, Value, Manager

from .generator import stream_producer, stream_consumer
from .estimator import run_naive_pca_estimator
from .temporal_neighbor import run_temporal_neighbors
from .outlying_attributes import run_outlying_attributes
from .mp_tasks import join_processes
from .mp_tasks import terminate_processes

from exos.explainer.temporal_neighbor import SequentialKMeans


import time
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)



def run_naive_exos_simulator(sources, d, k, attributes, feature_names, 
                             window_size, n_clusters = (), n_init_data = (), 
                             round_flag=True, threshold=0.0):
    """
    Parameters
    ----------
    sources : list
        list of the TemporalDataStream objects
    d : int
        total number of attributes in all streams
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
    n_init_data : tuple
        list of numpy array   
    
    Return
    a dictionary results

    results.keys()
    dict_keys(['output', 'simulator_time'])
        results['output'].keys()
        dict_keys(['window_0', 'window_1'])
            results['output']['window_0'].keys()
            dict_keys([0, 1, 2, 'est_time'])
                results['output']['window_0'][0].keys()
                dict_keys(['out_attrs', 'outlier_indices', 'temporal_neighbor_time', 'out_attrs_time'])
                    results['output']['window_0'][0]['out_attrs']
                        list of dictionary of feature names and their corresponding contribution value
                        the length of the list is equal to the number of outliers in the window ['window_0'] 
                        of the particular stream [0]
                    
                    results['output']['window_0'][0]['outlier_indices'].keys()
                    dict_keys([0, 1, 2]) --> we don't need info of other streams but 0
                    
                    results['output']['window_0'][0]['temporal_neighbor_time']
                    real number, running time required by temporal neighbor process at stream 0
                    
                    results['output']['window_0'][2]['out_attrs_time']
                    real number, running time required by outlying attributes at stream 0
        results['simulator_time']
        real number, running time required to run the entire windows
    """
    start = time.perf_counter()
    logging.info("Start exos simulator")
    n_streams = len(sources)

    ### initialize Value
    value = Value('i',n_streams)

    ### Initialize queues
    logging.info("Initializing Queues")
    manager = Manager()
    queues = [manager.Queue() for _ in range(n_streams)]
    
    buffer_queue = manager.Queue()
    buffer_queues = [manager.Queue() for _ in range(n_streams)]
    y_queue = manager.Queue()
    
    est_queues = [manager.Queue() for _ in range(n_streams)]
    est_time_queue = manager.Queue()
    neigh_queues = [manager.Queue() for _ in range(n_streams)]
    C_queues = [manager.Queue() for _ in range(n_streams)]
    exos_queues = [manager.Queue() for _ in range(n_streams)]

    ### Initialize conditions
    condition = Condition()
    exos_condition = Condition()
    neigh_condition = Condition()

    ### Start processes
    producers = [Process(target=stream_producer, 
                         args=(condition, queues[i], sources[i], i, window_size), 
                         daemon=True) for i in range(n_streams)]
    for p in producers:
        p.start()
    
    consumer = Process(target=stream_consumer,
                       args=(condition, queues, buffer_queue, buffer_queues, y_queue),
                       daemon=True)
    consumer.start()

    estimator_p = Process(target=run_naive_pca_estimator,
                        args=(value, neigh_condition, exos_condition, 
                              est_queues, est_time_queue, buffer_queue, 
                              n_streams, d, k, y_queue, attributes),
                        daemon=True)
    estimator_p.start()

    neighbors = list()
    for stream_id in range(n_streams):
        ncluster = 4
        if n_clusters:
            ncluster = n_clusters[stream_id]
        init_data = None 
        if n_init_data:
            init_data = n_init_data[stream_id]
        if stream_id < n_streams -1:
            n_attrs = attributes[stream_id+1] - attributes[stream_id]
        else:
            n_attrs = d - attributes[stream_id]

        clustering = SequentialKMeans(n_attrs)
        C_queues[stream_id].put(clustering)

        neighbor = Process(target=run_temporal_neighbors,
                           args=(neigh_condition, 
                                 neigh_queues[stream_id], 
                                 buffer_queues[stream_id], 
                                 stream_id,
                                 ncluster, 
                                 init_data,
                                 C_queues[stream_id]),
                           daemon=True)
        neighbors.append(neighbor)

    for neighbor in neighbors:
        neighbor.start()

    explanations = list()
    for stream_id in range(n_streams):
        explanation = Process(target=run_outlying_attributes,
                              args=(value, exos_condition, est_queues[stream_id], 
                                    neigh_queues[stream_id], exos_queues[stream_id], 
                                    stream_id, attributes, feature_names, 
                                    round_flag, threshold),
                              daemon=True)
        explanations.append(explanation)
    for explanation in explanations:
        explanation.start()

    join_processes(n_streams, producers, consumer, estimator_p, neighbors, explanations, value)

    result = {}
    counter = 0
    while True:
        outputs = {}
        for stream_id in range(n_streams):
            output = exos_queues[stream_id].get()
            if output is not None:
                outputs[stream_id] = output
        est_time = est_time_queue.get()
        if est_time is None:
            break
        outputs['est_time'] = est_time
        result[f'window_{counter}'] = outputs
        counter += 1

    logging.info('Terminating processes')
    terminate_processes(n_streams, producers, consumer, estimator_p, neighbors, explanations,
                        queues, buffer_queue, buffer_queues, est_queues, est_time_queue,
                        neigh_queues, exos_queues, y_queue, Q_queue=None)

    end = time.perf_counter()
    logging.info('Done')
    return {'output' : result, 'simulator_time' : end-start}