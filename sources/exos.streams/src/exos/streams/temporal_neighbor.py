from exos.explainer import temporal_neighbor
import numpy as np
import time
import os
import sys
import setproctitle

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

def run_temporal_neighbors(neigh_condition, neigh_queue, bqueue, stream_id, ncluster, init_data):
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
                clustering = temporal_neighbor.cluster_data(buffer, ncluster, init_data)
                inlier_centroids = [cluster.centroid for cluster in clustering.clusters]
                inlier_centroids = np.array(inlier_centroids)
                end = time.perf_counter()
                neigh_queue.put((inlier_centroids, end-start))
                with neigh_condition:
                    neigh_condition.wait()
                logging.info(f'Temporal neighbor {stream_id} woke')
        except Exception e:
            logging.error(f'Exception at temporal neighbor {stream_id} : {e}')
            
    logging.info(f'Temporal neighbor {stream_id} / {pid} exit')
    sys.stdout.flush()
        