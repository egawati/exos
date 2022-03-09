import numpy as np

def get_outliers(arr, y_queue, n_streams):
    y_d = y_queue.get()
    new_y_d = {}
    new_y_d[0] = list()
    outlier_index = set(y_d[0])    
    
    for i in range(1, n_streams):
        outlier_index.update(y_d[i])
        new_y_d[i] = list()
    outlier_index = list(outlier_index)
    
    for i in range(n_streams):
        for idx in y_d[i]:
            if idx in outlier_index:
                new_y_d[i].append(outlier_index.index(idx))
    
    outliers= np.take(arr, outlier_index, axis=0)
    return outliers, y_d, new_y_d 