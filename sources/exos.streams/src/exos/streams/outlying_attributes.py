from exos.explainer.outlying_attributes import find_outlying_attributes
import numpy as np
import time
import os
import sys
import setproctitle
np.set_printoptions(precision=2)
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

def run_outlying_attributes(value, exos_condition, est_queue, neigh_queue, 
                            exos_queue, stream_id, attributes, 
                            feature_names, round_flag=False, threshold=0.0):
    pid = os.getpid()
    setproctitle.setproctitle(f"Exos.OA{stream_id}")
    while True:
        try:
            start = time.perf_counter()
            estimator_result = est_queue.get()
            neigh_result = neigh_queue.get()
            if estimator_result is None or neigh_result is None:
                exos_queue.put(None)
                logging.info(f"OA {stream_id} DONE\n")
                break
            else:
                logging.info('-------------------')
                logging.info(f'Generating outlying attributes at {stream_id}\n')
                outliers, outliers_est, outlier_indices = estimator_result
                #print(f"outliers {outliers}\n")
                #print(f'outliers_est {outliers_est}\n')
                
                clustering, neigh_run_time = neigh_result
                inlier_centroids = list()
                cluster_counts = list()
                for cluster in clustering.clusters:
                    inlier_centroids.append(cluster.centroid)
                    cluster_counts.append(cluster.N)

                inlier_centroids = np.array(inlier_centroids)
                logging.info(f'temporal neighbors at stream {stream_id} are {inlier_centroids}')
                
                if outliers.shape[0] == 0:
                    exos_queue.put({"out_attrs" : None, 
                                "outlier_indices" : None, 
                                "temporal_neighbor_time" : neigh_run_time, 
                                "out_attrs_time" : time.perf_counter()-start})
                    with exos_condition:
                        with value.get_lock():
                            value.value -= 1
                        exos_condition.notify()
                    continue

                d = inlier_centroids.shape[1]
                outlying_attributes = list()
                logging.info(f'At stream {stream_id} Outliers are {outliers}\n')
                logging.info(f'Estimated values of outliers are {outliers_est}')
                
                for i, outlier in enumerate(outliers):
                    out_attributes = find_outlying_attributes( outlier, 
                                                               outliers_est[i,:],
                                                               inlier_centroids, 
                                                               cluster_counts,
                                                               d, 
                                                               feature_names[stream_id], 
                                                               round_flag, 
                                                               threshold)
                    outlying_attributes.append(out_attributes)
                end = time.perf_counter()
                exos_queue.put({"out_attrs" : outlying_attributes, 
                                "outlier_indices" : outlier_indices, 
                                "temporal_neighbor_time" : neigh_run_time, 
                                "out_attrs_time" : end-start})
                logging.info(f'outlying attributes at stream {stream_id} is {outlying_attributes}')
                logging.info('-------------------')
                with exos_condition:
                    with value.get_lock():
                        value.value -= 1
                    exos_condition.notify()
        except Exception as e:
            logging.error(f'Exception at OA {stream_id} : {e}')
    logging.info(f'OA {stream_id} / {pid} exit\n')
    sys.stdout.flush()
