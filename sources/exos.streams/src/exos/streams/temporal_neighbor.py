from exos.explainer import temporal_neighbor
import numpy as np
import time
import os
import sys
import setproctitle

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

def run_temporal_neighbors(neigh_condition, neigh_queue, bqueue, stream_id, ncluster, init_data, C_queue):
    pid = os.getpid()
    setproctitle.setproctitle(f"Exos.TemporalNeighbor{stream_id}")
    while True:
        start = time.perf_counter()
        try:
            logging.info(f'Run temporal neighbor {stream_id}\n')
            buffer = bqueue.get()
            if buffer is None:
                logging.info(f'Temporal neighbor {stream_id} DONE')
                neigh_queue.put(None)
                break
            else:
                clustering = C_queue.get()
                clustering.reset_clusters()
                for i in range(buffer.shape[0]):
                    x = buffer[i,:]
                    clustering.absorb_datum(x)
                #inlier_centroids = [cluster.centroid for cluster in clustering.clusters]
                #inlier_centroids = np.array(inlier_centroids)
                end = time.perf_counter()
                C_queue.put(clustering)
                neigh_queue.put((clustering, end-start))
                with neigh_condition:
                    neigh_condition.wait()
                logging.info(f'Temporal neighbor {stream_id} woke')
        except Exception as e:
            logging.error(f'Exception at temporal neighbor {stream_id} : {e}')
            
    logging.info(f'Temporal neighbor {stream_id} / {pid} exit')
    sys.stdout.flush()
        