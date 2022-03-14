from exos.explainer.outlying_attributes import find_outlying_attributes
import time

def run_outlying_attributes(exos_condition, est_queues, neigh_queues, 
                            exos_queues, stream_id, attributes, 
                            feature_names, round_flag=False, multiplier=10):
    exos_condition.acquire()
    while True:
        try:
            print(f'Run explainer {stream_id}')
            start = time.perf_counter()
            estimator_result = est_queues[stream_id].get()
            neigh_result = neigh_queues[stream_id].get()
            if estimator_result is None and neigh_result is None:
                return
            else:
                outliers, outliers_est, new_y_d = estimator_result
                inlier_centroids, neigh_run_time = neigh_result
                print(f'inlier_centroids at {stream_id} is {inlier_centroids}')
                print(f'outliers at {stream_id} is {outliers}')
                print(f'outliers_est at {stream_id} is {outliers_est}')

                d = inlier_centroids.shape[1]
                outlying_attributes = list()
                # for i, outlier in enumerate(outliers):
                #     out_attributes = find_outlying_attributes( outlier, 
                #                                                outliers_est[i],
                #                                                inlier_centroids, 
                #                                                d, 
                #                                                feature_names, 
                #                                                round_flag, 
                #                                                multiplier)
                #     outlying_attributes.append(out_attributes)
                end = time.perf_counter()
                exos_queues[stream_id].put((outlying_attributes, new_y_d, neigh_run_time, end-start))
                with exos_condition:
                    exos_condition.notify_all()
        except:
            val = exos_condition.wait(5) # wait for 2 seconds
            if val:
                continue
            else:
                print(f'waiting timeout')
                break;
    exos_condition.release()