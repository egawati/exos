from exos.explainer.outlying_attributes import find_outlying_attributes
import time

def run_outlying_attributes(exos_condition, est_queue, neigh_queue, 
                            exos_queue, stream_id, attributes, 
                            feature_names, round_flag=False, multiplier=10):
    while True:
        try:
            start = time.perf_counter()
            estimator_result = est_queue.get()
            neigh_result = neigh_queue.get()
            if estimator_result is None and neigh_result is None:
                return
            else:
                outliers, outliers_est, new_y_d = estimator_result
                inlier_centroids, neigh_run_time = neigh_result
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
                exos_queue.put((outlying_attributes, new_y_d, neigh_run_time, end-start))
                with exos_condition:
                    exos_condition.notify_all()
        except neigh_queue.empty():
            pass