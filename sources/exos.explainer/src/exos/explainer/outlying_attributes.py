import numpy as np

from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

def generate_outlier_class(est_outlier, outlier, cluster_count, d, radius, round_flag=False):
    """
    Parameters
    ----------
    inlier_class : numpy array
        n x d numpy array 
    outlier: numpy array
        the outlier object
    d: int
        number of attributes
    round_flag : boolean
        whether to round each generated sampling
    multiplier: int
        determine the number of sampling to generate
    """
    dist = np.linalg.norm(est_outlier-outlier)
    if dist < radius:
        dist = radius
    covariance = np.identity(d) * dist / (3*2)
    n = d * cluster_count
    gaussian_data = np.random.multivariate_normal(outlier, covariance, n)
    if round_flag:
        gaussian_data = np.round(gaussian_data)
    outlier_class = np.vstack((gaussian_data, outlier))
    return outlier_class

def generate_inlier_class(est_outlier, inlier_centers, cluster_counts, d, round_flag=False):
    min_dist = np.linalg.norm(est_outlier-inlier_centers[0,:])
    inlier_nearest_neighbor = inlier_centers[0,:]
    idx = 0
    for i in range(1, inlier_centers.shape[0]):        
        dist = np.linalg.norm(est_outlier-inlier_centers[i,:])
        if min_dist < dist:
            min_dist = dist
            inlier_nearest_neighbor = inlier_centers[0,:]
            idx = i

    #logging.info(f'inlier {inlier_nearest_neighbor.shape}')
    covariance = np.identity(d) * min_dist / (3*2)
    #logging.info(f'd {d}')
    n = d * cluster_counts[idx]
    gaussian_data = np.random.multivariate_normal(inlier_nearest_neighbor, covariance, n)
    if round_flag:
        gaussian_data = np.round(gaussian_data)
    inlier_class = np.vstack((gaussian_data, est_outlier))
    return inlier_class, min_dist, cluster_counts[idx]


def compute_attribute_contribution(n_features, classifier):
    print(f'hyperplane weights are {classifier.coef_[0]}\n')
    abs_weights = np.abs(classifier.coef_[0])
    attr_contributions = abs_weights/np.sum(abs_weights)
    return attr_contributions


def map_feature_scores(feature_names, feature_scores, threshold=0.0):
    result = {k: v for k, v in zip(feature_names, feature_scores)}
    result = dict((k, v) for k, v in result.items() if v > threshold)
    #result = dict(sorted(result.items(), key=lambda result: result[1], reverse=True))
    return result

@ignore_warnings(category=ConvergenceWarning)
def run_svc(outlier_class, inlier_class, 
            regularization = 'l1', 
            regularization_param = 1, 
            intercept_scaling = 1):
    X = np.vstack((outlier_class, inlier_class))
    y = np.hstack((np.ones(outlier_class.shape[0]), np.zeros(inlier_class.shape[0])))
    dual = False
    clf = LinearSVC(penalty=regularization,
                    C=regularization_param,
                    dual=dual,
                    intercept_scaling=intercept_scaling)
    
    clf.fit(X, y)
    return clf

def find_outlying_attributes(outlier_point, est_outlier, 
                             inlier_centroids, cluster_counts, 
                             d, feature_names, 
                             round_flag=False, 
                             threshold=0.0):
    """
    Parameters
    ----------
    outlier_point: numpy array
    inlier_centroids : numpy array
        n x d numpy array of inlier centroids, 
        where n is the number of clusters and d is the number of features
    d: int 
        number  of attributes
    round_flag
        whether to round each generated sampling
    """
    inlier_class,min_dist, cluster_count = generate_inlier_class(est_outlier, inlier_centroids, cluster_counts,d, round_flag)
    outlier_class = generate_outlier_class(est_outlier, outlier_point, cluster_count, d, min_dist, round_flag)
    classifier = run_svc(outlier_class, inlier_class)
    #feature_scores = compute_simple_feature_contribution(d, (classifier,))
    attr_contributions = compute_attribute_contribution(d, classifier)
    result = map_feature_scores(feature_names, attr_contributions, threshold)
    return result