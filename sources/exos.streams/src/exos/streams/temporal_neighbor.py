from exos.explainer import temporal_neighbor
import numpy as np
import time

def run_temporal_neighbors(exos_condition, neigh_queues, buffer_queues, stream_id, ncluster, init_data):
    while True:
        print(f"Run temporal_neighbor at {stream_id}")
        start = time.perf_counter()
        buffer = buffer_queues[stream_id].get()
        if buffer is None:
            return
        else:    
            clustering = temporal_neighbor.cluster_data(buffer, ncluster, init_data)
            inlier_centroids = [cluster.centroid for cluster in clustering.clusters]
            inlier_centroids = np.array(inlier_centroids)
            end = time.perf_counter()
            neigh_queues[stream_id].put((inlier_centroids, end-start))

            with exos_condition:
                exos_condition.wait()