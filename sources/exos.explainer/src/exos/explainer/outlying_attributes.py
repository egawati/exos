import numpy as np

from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

def generate_outlier_class(inlier_centers, outlier, d, round_flag=False, multiplier = 10):
    """
    Parameters
    ----------
    inlier_centers : numpy array
        n x d numpy array of microclusters' centers, 
        where n is the number of nearby clusters and d is the number of features
    outlier: numpy array
        the outlier object
    d: int
        number of attributes
    round_flag : boolean
        whether to round each generated sampling
    multiplier: int
        determine the number of sampling to generate
    """
    neighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(inlier_centers)
    neighbor_min_dist = neighbors.kneighbors([outlier])[0][0, 0] / d
    covariance = np.identity(d) * neighbor_min_dist / (3**2)
    n = inlier_centers.shape[0]
    if n < inlier_centers.shape[1]:
        n = inlier_centers.shape[1]
    n = n * multiplier
    gaussian_data = np.random.multivariate_normal(outlier, covariance, n)
    if round_flag:
        gaussian_data = np.round(gaussian_data)
    outlier_class = np.vstack((gaussian_data, outlier))
    return outlier_class, neighbors, neighbor_min_dist

def compute_feature_contribution(n_features, npoints, classifiers):
    """
    Parameters
    ----------
    n_features
        number of features
    n_points
        list of number of inlier class used by each classifiers
    classifiers
        the classifier models
    """
    feature_scores = np.zeros(n_features)
    for npoint, clf in zip(npoints, classifiers):
        feature_scores += npoint * np.abs(clf.coef_[0])
    feature_scores /= float(np.sum(npoints))
    feature_scores /= np.sum(feature_scores)
    return feature_scores


def compute_simple_feature_contribution(n_features, classifiers):
    """
    Parameters
    ----------
    n_features
        number of features
    classifiers
        the classifier models
    """
    feature_scores = np.zeros(n_features)
    for clf in classifiers:
        feature_scores += np.abs(clf.coef_[0])
    feature_scores = feature_scores / len(classifiers)
    return feature_scores


def map_feature_scores(feature_names, feature_scores):
    result = {k: v for k, v in zip(feature_names, feature_scores)}
    return result


def run_svc(outlier_class, inlier_class, 
            regularization = 'l1', 
            regularization_param = 1, 
            intercept_scaling = 1):
    n_samples = outlier_class.shape[0] + inlier_class.shape[0]
    dual = False
    clf = LinearSVC(penalty=regularization,
                    C=regularization_param,
                    dual=dual,
                    intercept_scaling=intercept_scaling)
    X = np.vstack((outlier_class, inlier_class))
    y = np.hstack((np.ones(outlier_class.shape[0]), np.zeros(inlier_class.shape[0])))
    clf.fit(X, y)
    return clf

def find_outlying_attributes(outlier_point, inlier_centroids, d, feature_names, round_flag=False, multiplier=10):
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
    outlier_class, neighbors, neighbor_min_dist = generate_outlier_class(inlier_centroids, 
                                                                         outlier_point, 
                                                                         round_flag, 
                                                                         multiplier)
    inlier_nearest_neighbor = neighbors.kneighbors(outlier_point)
    covariance = np.identity(d) * neighbor_min_dist / (3**2)
    n = d * multiplier
    gaussian_data = np.random.multivariate_normal(outlier, covariance, n)
    if round_flag:
        gaussian_data = np.round(gaussian_data)
    inlier_class = np.vstack((gaussian_data, inlier_nearest_neighbor))

    classifier = run_svc(outlier_class, inlier_class)
    feature_scores = compute_simple_feature_contribution(d, (classifier,))
    return map_feature_scores(feature_names, feature_scores)
    
