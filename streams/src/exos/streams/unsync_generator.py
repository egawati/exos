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

def csv_to_stream(filepath, 
				  arrival_rate= 'Fixed',
				  time_unit = 'seconds',
				  time_interval = 1,
				  delay_time = 1,
				  delay_unit = 'milliseconds',
				  label='label'):
	raw = pd.read_csv(filepath)
	dataX = None
	datay = None
	if label:
		dataX = raw.drop(['label'], axis=1)
		datay = raw[['label']].values
	else:
		dataX = raw
	features = tuple(dataX.columns)
	dataX = dataX.values
	n_features = dataX.shape[1]
	time = generate_timestamp(dataX.shape[0],
                            arrival_rate=arrival_rate,
                            time_unit=time_unit,
                            time_interval=time_interval)
	delay_time = np.timedelta64(delay_time, delay_unit)
	tstream = TemporalDataStream(dataX, datay, time, sample_delay=delay_time, ordered=True)
	return tstream, n_features

def check_tuple_value(tuple_, idx):
	if not tuple_:
		return None
	else:
		return tuple_[idx]

def multiple_csv_to_streams(filepaths=(), 
				  arrival_rates = (),
				  time_units = 'seconds',
				  time_intervals = (),
				  delay_times = (),
				  delay_units = 'milliseconds',
				  labels = ()):
	sources = list()
	for i, filepath in enumerate(filepaths):
		arrival_rate = arrival_rates[i] if check_tuple_value(arrival_rates, i) is not None else 'Fixed'
		time_interval = time_intervals[i] if check_tuple_value(time_intervals, i) is not None else 1
		delay_time = delay_times[i] if check_tuple_value(delay_times, i) is not None else 1
		label = labels[i] if check_tuple_value(labels, i) is not None else False
		tstream, n_features = csv_to_stream(filepath,
											arrival_rate,
											time_unit,
											time_interval,
											delay_time,
											delay_unit,
											label)
		sources.append((tstream, n_features))
	return sources

def stream_producer(condition, queue, source, source_id, window_size, arrival_rate):
    setproctitle.setproctitle(f"Exos.producer{source_id}")
    while True:
        if not source.has_more_samples():
        	queue.put(None)
        	break 
        else:
            X, y, _, _, _ = source.next_sample(window_size//arrival_rate)
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