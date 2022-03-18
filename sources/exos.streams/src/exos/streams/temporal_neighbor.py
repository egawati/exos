from exos.explainer import temporal_neighbor
import numpy as np
import time
import os
import sys
import setproctitle

def run_temporal_neighbors(neigh_condition, neigh_queue, bqueue, stream_id, ncluster, init_data):
    pid = os.getpid()
    setproctitle.setproctitle(f"Exos.TemporalNeighbor{stream_id}")
    while True:
        start = time.perf_counter()
        try:
            buffer = bqueue.get()
            if buffer is None:
                print(f'Temporal neighbor {stream_id} DONE')
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
                print(f'Temporal neighbor {stream_id} woke')
        except bqueue.Empty:
            pass
    print(f'Temporal neighbor {stream_id} / {pid} exit')
    sys.stdout.flush()
        