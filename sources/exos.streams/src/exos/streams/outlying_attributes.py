from exos.explainer.outlying_attributes import find_outlying_attributes

def run_outlying_attributes(exos2_queue, stream_id, n_streams,
                            all_outliers, all_outliers_est, outlier_stream_idx,
                            inlier_centroids, attributes, feature_names, 
                            round_flag=False, multiplier=10, profiling=False):
    
    outlier_index = outlier_stream_idx[stream_id]
    start_idx = attributes[stream_id]
    
    if stream_id < n_streams - 1:
        end_idx = attributes[stream_id+1]
        outliers = np.take(all_outliers[:,start_idx:end_idx], outlier_index, axis=0)
        outliers_est = np.take(all_outliers_est[:,start_idx:end_idx], outlier_index, axis=0)
    else:
        outliers = np.take(all_outliers[:,start_idx:], outlier_index, axis=0)
        outliers_est = np.take(all_outliers_est[:,start_idx:], outlier_index, axis=0)

    d = inlier_centroids.shape[1]
    outlying_attributes = list()
    for i, outlier in enumerate(outliers):
        out_attributes = find_outlying_attributes( outlier, 
                                                   outliers_est[i],
                                                   inlier_centroids, 
                                                   d, 
                                                   feature_names, 
                                                   round_flag, 
                                                   multiplier)
        outlying_attributes.append(out_attributes)
    exos2_queue.put((stream_id, outlying_attributes))