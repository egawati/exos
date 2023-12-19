import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def join_processes(n_streams, producers, consumer, estimator_p, neighbors, explanations, value):
    for stream_id in range(n_streams):
        producers[stream_id].join()
        logging.info(f'produser at main {stream_id} done\n')
    
    consumer.join()
    logging.info('consumer at main done\n')
    logging.info(f'value is {value.value}\n')
    
    for stream_id in range(n_streams):
        neighbors[stream_id].join()
        logging.info(f'temporal neighbor at main {stream_id} done\n')
    logging.info(f'value is {value.value}\n')
        
    estimator_p.join()
    logging.info('estimator at main done\n')
    logging.info(f'value is {value.value}\n')
        
    for stream_id in range(n_streams):
        explanations[stream_id].join()
        logging.info(f'OA at main {stream_id} done\n')

    logging.info(f'value is {value.value}\n')
    

def terminate_processes(n_streams, producers, consumer, estimator_p, neighbors, explanations,
                        queues, buffer_queue, buffer_queues, est_queues, est_time_queue,
                        neigh_queues, exos_queues, y_queue, Q_queue=None):
    for stream_id in range(n_streams):
        producers[stream_id].terminate()
        queues[stream_id]._close()
    
    consumer.terminate()
    buffer_queue._close()
    y_queue._close()
    
    #estimator_p.terminate()
    est_time_queue._close()

    if Q_queue is not None:
        Q_queue._close()
    
    for stream_id in range(n_streams):
        buffer_queues[stream_id]._close()
        neighbors[stream_id].terminate()
        neigh_queues[stream_id]._close()
        
        est_queues[stream_id]._close()
        
        explanations[stream_id].terminate()
        exos_queues[stream_id]._close()
