from exos.explainer import temporal_neighbor
import numpy as np
import time

def run_temporal_neighbors(exos_condition, neigh_queue, bqueue, stream_id, ncluster, init_data):
    while True:
        start = time.perf_counter()
        try:
            buffer = bqueue.get()
            if buffer is None:
                return
            else:
                clustering = temporal_neighbor.cluster_data(buffer, ncluster, init_data)
                inlier_centroids = [cluster.centroid for cluster in clustering.clusters]
                inlier_centroids = np.array(inlier_centroids)
                end = time.perf_counter()
                neigh_queue.put((inlier_centroids, end-start))
                with exos_condition:
                    exos_condition.wait()
        except bqueue.empty():
            pass
            
        