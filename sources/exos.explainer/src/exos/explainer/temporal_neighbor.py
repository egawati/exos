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
    def __init__(self, d, k=8, init_data=None):
        """
        k : number of clusters
        d : number of attributes
        init_data : when it is not None, it is a numpy array of size n x d
        """
        self.k  = k
        self.d = d
        self.clusters = [None] * k
        if init_data is None:
            for i in range(k):
                centroid = np.random.rand(d)
                self.clusters[i] = ClusterKMeans(centroid, 0)
        else:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(init_data)
            centroids = kmeans.cluster_centers_
            for i in range(k):
                centroid = centroids[i]
                self.clusters[i] = ClusterKMeans(centroid, 0)
        self.y = list()
    
    def absorb_incoming_datum(self, x, label=False):
        idx = 0
        min_dist = np.linalg.norm(self.clusters[idx].centroid- x)
        for i in range(1, self.k):
            dist = np.linalg.norm(self.clusters[i].centroid- x)
            if dist < min_dist:
                idx = i
                min_dist = dist
        self.clusters[idx].update_cluster_feature(x)
        if label:
            self.y.append(idx)

def cluster_data(X, k=8, init_data=None):
	"""
	X : numpy array of n x d
	k : number of clusters
	init_data : when it is not None, it is a numpy array of size n x d
	"""
	d = X.shape[1]
	clustering = SequentialKMeans(d, k, init_data)
	for i in range(X.shape[0]):
	    x = X[i,:]
	    clustering.absorb_incoming_datum(x, label=True)
	return clustering