import numpy as np
import pandas as pd
from skmultiflow.data import TemporalDataStream

from threading import Thread, Condition, Lock

from queue import Queue

from exos.explainer.estimator import dbpca

from .generator import stream_producer, stream_consumer
from .estimator import run_dbpca_estimator
from .temporal_neighbor import run_temporal_neighbors
from .outlying_attributes import run_outlying_attributes

import time

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


class EThread(Thread):
    def __init__(self, target, *args):
        Thread.__init__(self, target=target, args=args)
        self.start()

def join_threads(n_streams, producers, consumer, estimator_p, neighbors, explanations, value):
    for stream_id in range(n_streams):
        producers[stream_id].join()
        print(f'produser at main {stream_id} done')
    consumer.join()
    print('consumer at main done')

    for stream_id in range(n_streams):
        if value.value == -1:
            neighbors[stream_id].terminate()
            print(f'temporal neighbor at main {stream_id} done') 
    
    estimator_p.join()
    print('estimator at main done')    
    
    for stream_id in range(n_streams):
        explanations[stream_id].join()
        print(f'explanation at main {stream_id} done')
    

def terminate_threads(n_streams, producers, consumer, estimator_p, neighbors, explanations,
                        queues, buffer_queue, buffer_queues, est_queues, est_time_queue,
                        neigh_queues, exos_queues, y_queue, Q_queue):
    for stream_id in range(n_streams):
        #producers[stream_id].terminate()
        queues[stream_id].close()
    
    #consumer.terminate()
    buffer_queue.close()
    y_queue.close()
    
    #estimator_p.terminate()
    est_time_queue.close()
    Q_queue.close()
    
    for stream_id in range(n_streams):
        buffer_queues[stream_id].close()
        #neighbors[stream_id].terminate()
        neigh_queues[stream_id].close()
        
        est_queues[stream_id].close()
        
        #explanations[stream_id].terminate()
        exos_queues[stream_id].close()


from .constant import value

def run_exos_simulator(sources, d, k, attributes, feature_names, 
                       window_size, n_clusters = (), n_init_data = (), 
                       multiplier = 10, round_flag=True):
    """
    Parameters
    ----------
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
    n_init_data : tuple
        list of numpy array   
    multiplier : int
        the number of data points to sample when creating inlier/outlier class

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
                    real number, running time required by temporal neighbor Thread at stream 0
                    
                    results['output']['window_0'][2]['out_attrs_time']
                    real number, running time required by outlying attributes at stream 0
        results['simulator_time']
        real number, running time required to run the entire windows
    """
    global value

    start = time.perf_counter()
    logging.info("Start exos simulator")
    n_streams = len(sources)

    value.set_initial_value(n_streams)

    ### Initialize queues
    logging.info("Initializing Queues")
    queues = [Queue() for _ in range(n_streams)]
    
    buffer_queue = Queue()
    buffer_queues = [Queue() for _ in range(n_streams)]
    y_queue = Queue()
    
    est_queues = [Queue() for _ in range(n_streams)]
    est_time_queue = Queue()
    neigh_queues = [Queue() for _ in range(n_streams)]
    Q_queue = Queue()
    exos_queues = [Queue() for _ in range(n_streams)]


    Q = dbpca.initialize_Q(d,k)
    Q_queue.put(Q)

    ### Initialize conditions
    condition = Condition()
    exos_condition = Condition()
    neigh_condition = Condition()

    ### Start threads
    producers = [  Thread(target=stream_producer, 
                         args=(condition, queues[i], sources[i], i, window_size), 
                         daemon=True) for i in range(n_streams)]
    for p in producers:
        p.start()
    
    consumer = Thread(target=stream_consumer,
                       args=(condition, queues, buffer_queue, buffer_queues, y_queue),
                       daemon=True)
    consumer.start()

    estimator_p = Thread(target=run_dbpca_estimator,
                        args=(value, neigh_condition, exos_condition, 
                              est_queues, est_time_queue, buffer_queue, 
                              n_streams, Q_queue, d, k, y_queue, attributes),
                        daemon=True)
    estimator_p.start()

    neighbors = list()
    for stream_id in range(n_streams):
        ncluster = 2
        if n_clusters:
            ncluster = n_clusters[stream_id]
        init_data = None 
        if n_init_data:
            init_data = n_init_data[stream_id]
        neighbor = Thread(target=run_temporal_neighbors,
                           args=(neigh_condition, 
                                 neigh_queues[stream_id], 
                                 buffer_queues[stream_id], 
                                 stream_id,
                                 ncluster, 
                                 init_data),
                           daemon=True)
        neighbors.append(neighbor)

    for neighbor in neighbors:
        neighbor.start()

    explanations = list()
    for stream_id in range(n_streams):
        explanation = Thread(target=run_outlying_attributes,
                              args=(value, exos_condition, est_queues[stream_id], 
                                    neigh_queues[stream_id], exos_queues[stream_id], 
                                    stream_id, attributes, feature_names, 
                                    round_flag, multiplier),
                              daemon=True)
        explanations.append(explanation)
    for explanation in explanations:
        explanation.start()

    join_threads(n_streams, producers, consumer, estimator_p, neighbors, explanations, value)

    result = {}
    counter = 0
    while True:
        outputs = {}
        for stream_id in range(n_streams):
            output = exos_queues[stream_id].get()
            outputs[stream_id] = output
        est_time = est_time_queue.get()
        if est_time is None:
            break
        outputs['est_time'] = est_time
        result[f'window_{counter}'] = outputs
        counter += 1

    end = time.perf_counter()
    logging.info('Done')
    return {'output' : result, 'simulator_time' : end-start}