
import numpy as np
import time
import math
from sklearn.cluster import KMeans

class ClusterKMeans():
    def __init__(self, centroid, N):
        self.centroid = centroid
        self.N = 0
    
    def update_cluster_feature(self, x):
        self.N += 1
        self.centroid = self.centroid + (x-self.centroid)/self.N
    
    def reset_cluster_feature(self):
        self.N = 0

class SequentialKMeans:
    def __init__(self, d, init_centroids=None, l=8, init_data=None):
        """
        k : number of clusters
        d : number of attributes
        init_data : when it is not None, it is a numpy array of size n x d
        """
        self.k  = l
        self.d = d
        self.clusters = [None] * self.k
        if init_data is None:
            for i in range(self.k):
                centroid = init_centroids[i]
                self.clusters[i] = ClusterKMeans(centroid, 0)
        else:
            init_centroids = init_data[0:l,:]
            logging.info(f'init centroids are {init_centroids}')
            kmeans = KMeans(n_clusters=self.k, init=init_centroids).fit(init_data)
            centroids = kmeans.cluster_centers_
            for i in range(self.k):
                centroid = centroids[i]
                self.clusters[i] = ClusterKMeans(centroid, 0)
        self.y = list()
    
    def absorb_datum(self, x, label=False):
        idx = 0
        min_dist = np.linalg.norm(self.clusters[idx].centroid - x)
        for i in range(1, self.k):
            dist = np.linalg.norm(self.clusters[i].centroid - x)
            if dist < min_dist:
                idx = i
                min_dist = dist
        self.clusters[idx].update_cluster_feature(x)
        if label:
            self.y.append(idx)

    def reset_clusters(self):
        for i in range(self.k):
            self.clusters[i].N = 0


def cluster_data(X, l=8, init_data=None):
	"""
	X : numpy array of n x d
	k : number of clusters
	init_data : when it is not None, it is a numpy array of size n x d
	"""
	d = X.shape[1]
	clustering = SequentialKMeans(d, l, init_data)
	for i in range(X.shape[0]):
	    x = X[i,:]
	    clustering.absorb_datum(x, label=True)
	return clustering