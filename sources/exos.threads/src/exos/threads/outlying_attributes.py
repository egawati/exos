from exos.explainer.outlying_attributes import find_outlying_attributes
import time
import os
import sys
import setproctitle

from .constant import value

def run_outlying_attributes(value, exos_condition, est_queue, neigh_queue, 
                            exos_queue, stream_id, attributes, 
                            feature_names, round_flag=False, multiplier=10):
    pid = os.getpid()
    while True:
        try:
            start = time.perf_counter()
            estimator_result = est_queue.get()
            neigh_result = neigh_queue.get()
            if estimator_result is None or neigh_result is None:
                exos_queue.put(None)
                print(f"OA {stream_id} DONE\n")
                break
            else:
                print(f'Generating outlying attributes at {stream_id}\n')
                outliers, outliers_est, new_y_d = estimator_result
                inlier_centroids, neigh_run_time = neigh_result
                
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
                for i, outlier in enumerate(outliers):
                    out_attributes = find_outlying_attributes( outlier, 
                                                               outliers_est[i,:],
                                                               inlier_centroids, 
                                                               d, 
                                                               feature_names, 
                                                               round_flag, 
                                                               multiplier)
                    outlying_attributes.append(out_attributes)
                end = time.perf_counter()
                exos_queue.put({"out_attrs" : outlying_attributes, 
                                "outlier_indices" : new_y_d, 
                                "temporal_neighbor_time" : neigh_run_time, 
                                "out_attrs_time" : end-start})
                with exos_condition:
                    value.decrement()
                    exos_condition.notify_all()
        except Exception as e:
            print(f'Exception at OA {stream_id} : {e}')
    print(f'OA {stream_id} / {pid} exit\n')
    sys.stdout.flush()
