import numpy as np
import pandas as pd
from skmultiflow.data import TemporalDataStream
from multiprocessing import Process, Queue, Event, Condition, Manager

from .producer import multiple_csv_to_streams

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def concatenate_buffers(d, n_streams):
    n_points = d[0].shape[0] #the number of data points in each streams
    arr = None
    for i in range(n_points):
        new_point = d[0][i]
        for j in range(1, n_streams):
            new_point = np.concatenate((new_point, d[j][i]))
        if arr is None:
            arr = new_point
        else:
            arr = np.vstack((arr, new_point))
    return arr


def run_experiment(buffer_queue, experiment_queue, n_streams):
    """
    call EXOS here
    """
    while True:
        d = buffer_queue.get()
        if d is not None:
            arr = concatenate_buffers(d, n_streams)
            experiment_queue.put(arr)
        else:
            experiment_queue.put(d)
            return            

if __name__ == '__main__':
    data1 = np.array([[1,2],[3,4],[5,6],[7,8], [9,10]])
    data2 = np.array([[11,12,101],[13,14,102],[15,16,103],[17,18,104], [19,20,105]])
    data3 = np.array([[21,22],[23,24],[25,26],[27,28], [29,30]])
    
    y = np.array([100, 200, 300, 400, 500])
    ts1 = TemporalDataStream(data1, y)
    ts2 = TemporalDataStream(data2, y)
    ts3 = TemporalDataStream(data3, y)

    sources = (ts1, ts2, ts3)
    n_streams = len(sources)
    queues = [Queue()] * n_streams
    
    condition = Condition()

    producers = [Process(target=stream_producer, args=(condition, queues, sources[i], i, 2), daemon=True) for i in range(n_streams)]
    for p in producers:
        p.start()
    
    buffer_queue = Queue()        
    consumer = Process(target=stream_consumer, args=(condition, queues, buffer_queue), daemon=True)
    consumer.start()

    experiment_queue = Queue()
    experiment = Process(target=run_experiment, args=(buffer_queue, experiment_queue, n_streams), daemon=True)
    experiment.start()

    for p in producers:
        p.join()

    for queue in queues:
        queue.put(None)
    consumer.join()

    buffer_queue.put(None)
    while True:
        result = experiment_queue.get()
        if result is None:
            break
        else:
            print(result)
    experiment_queue.put(None)
    experiment.join()

    print("Done")