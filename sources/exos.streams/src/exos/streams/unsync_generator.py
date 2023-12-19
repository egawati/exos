import numpy as np
import pandas as pd
import os
import sys

from skmultiflow.data import TemporalDataStream

from sklearn import preprocessing

from .utils import time_unit_numpy
from .utils import generate_timestamp
import setproctitle
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def stream_producer(condition, queue, source, source_id, window_size, arrival_rates=1):
    setproctitle.setproctitle(f"Exos.producer{source_id}")
    while True:
        if not source.has_more_samples():
        	queue.put(None)
        	break 
        else:
            X, y, _, _, _ = source.next_sample(window_size/arrival_rate)
            queue.put((X, y, source_id))
            with condition:
                condition.wait()
    logging.info(f'producer {source_id} / {os.getpid()} exit')
    sys.stdout.flush()

def stream_consumer(condition, queues, buffer_queue, buffer_queues, y_queue, normalized=False):
	"""
	condition: mp.Condition
	queues: list of Queues (of length n_stream)
		used to get data from stream_producer
	buffer_queue: Queue
		used to put data sent to the estimator process
	buffer_queues: list of Queues (of length n_stream)
		used to put data to send to the temporal neighbor proceses
	y_queue: queue
		used to put info about outliers 
	"""
	hash_d = {}
	y_d = {}
	exit = False
	setproctitle.setproctitle("Exos.consumer")
	while not exit:
	    results = [queue.get() for queue in queues]
	    for result in results:
	        if result is None:
	        	for bqueue in buffer_queues:
	        		bqueue.put(None)
	        	buffer_queue.put(None)
	        	y_queue.put(None)
	        	with condition:
	        		condition.notify_all()
	        	exit = True
	        	break
	    if (exit):
	    	break
	    for X, y, source_id in results:
	    	if normalized:
	    		scaler = preprocessing.StandardScaler().fit(X)
	    		new_X = scaler.transform(X)
	    	else:
	    		new_X = X
	    	hash_d[source_id] = new_X
	    	### assuming we have run outlier detection
	    	### a data point is an outlier is = 1
	    	y_d[source_id] = np.where(y==1)[0]
	    	buffer_queues[source_id].put((new_X,y))
	        #if source_id == 1:
	        #S	print(f'stream 1 {y_d[source_id]}')
	    buffer_queue.put(hash_d)
	    y_queue.put(y_d)
	    with condition:
	        condition.notify_all()
	logging.info(f'customer {os.getpid()} exit')
	sys.stdout.flush()