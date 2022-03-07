import numpy as np
import pandas as pd

from skmultiflow.data import TemporalDataStream

from .utils import time_unit_numpy
from .utils import generate_timestamp

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

def stream_producer(condition, queues, source, source_id, window_size):
    while True:
        if not source.has_more_samples():
            return 
        else:
            X, y, _, _, _ = source.next_sample(window_size)
            queues[source_id].put((X, y, source_id))
            #print("Produced {} = {}".format(source_id, X))
            with condition:
                condition.wait()
        
def stream_consumer(condition, queues, buffer_queue, y_queue):
    hash_d = {}
    y_d = {}
    while True:
        results = [queue.get() for queue in queues]
        for result in results:
            if result is None:
                return
        for X, y, source_id in results:
            hash_d[source_id] = X
            ### assuming we have run outlier detection
            ### a data point is an outlier is = 1
            y_d[source_id] = np.where(y==1)[0] 
        buffer_queue.put(hash_d)
        y_queue.put(y_d)
        with condition:
            condition.notify_all()