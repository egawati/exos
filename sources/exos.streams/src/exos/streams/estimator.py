from exos.explainer.estimator import dbpca
from .common import get_outliers

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def concatenate_buffers(hash_d, n_streams):
    """
    Combine data points from each stream on a period of time
    Assumption: each stream produces the same number of data points
    Parameters
    ----------
    hash_d : dict
        number of attributes
    n_streams : int
        number of principle component
    Returns
    -------
    arr : np.array
        numpy array of shape n_points x total number of attributes
    -------
    """
    n_points = hash_d[0].shape[0] #the number of data points in each streams
    arr = None
    for i in range(n_points):
        new_point = hash_d[0][i]
        for j in range(1, n_streams):
            new_point = np.concatenate((new_point, hash_d[j][i]))
        if arr is None:
            arr = new_point
        else:
            arr = np.vstack((arr, new_point))
    return arr

def run_dbpca_estimator(est_queue, hash_d, n_streams, Q_queue, d, k, y_queue, profiling=False):
    """
    Used when dbpca run in multiprocessing setting
    Parameters
    ----------
    est_queue: Queue()
        used to return values
    hash_d: dict
        buffer for each stream
        Example: two streams: stream 0 (2 features) and 1 (2 features), each has 5 data points 
        {
         0: array([[ 100,   90],
                   [1000,  100],
                   [  25,   26],
                   [  27,   28],
                   [  29,   30]]), 
         1: array([[  22,   22],
                    [1000, 1000],
                    [ 225,  226],
                    [ 227,  228],
                    [ 229,  230]])
        }
    n_streams: int
        number of streams
    Q_queue: Queue()
        store Q (estimated eigen vectors) 
    d: int
        total number of features (sum of number of attributes from each stream)
    k: int
    y_queue: Queue()
        store information about detected outlier on current window
    profiling: boolean
        default value: False
    """
    if profiling:
        start = time.perf_counter() #start measuring estimation function
    arr = concatenate_buffers(hash_d, n_streams)
    W = arr.T # d x m 
    Q = Q_queue.get() # d x k 
    Q = dbpca.update_Q(W,d,k,Q) # d x k 
    
    outliers, y_d, new_y_d = get_outliers(arr, y_queue, n_streams) ## numpy array of n_outliers x d
    Y = outliers.dot(Q) # m x k
    outliers_est = Y.dot(Q.T) # m x d
    
    Q_queue.put(Q)
    
    end = time.perf_counter() #end measuring estimation function
    
    logging.info(f"Running DBPCA estimation function in {end-start} second")
    est_queue.put((outliers, outliers_est, new_y_d, y_d, end-start))