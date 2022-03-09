from exos.explainer import temporal_neighbor

def run_temporal_neighbors(neigh_queues, buffer, stream_id, ncluster, init_data, profiling=False):
    if profiling:
        start = time.perf_counter()
    clustering = temporal_neighbor.cluster_data(buffer, ncluster, init_data)
    if profiling:
        end = time.perf_counter()
        print(f"Running temporal neighbor function in {end-start} second")
        neigh_queues[stream_id].put((stream_id, clustering, end-start))
    else:
        neigh_queues[stream_id].put((stream_id, clustering))