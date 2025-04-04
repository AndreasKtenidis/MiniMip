import numpy as np

from data.numpy_dataset.fed_multiset import Multiset
from function.abstract_function import AggFunc
from data.numpy_dataset.fed_array import FedArray

class KMeans(AggFunc):

    def __init__(self):
        self.x = None
        self.centroids = None
        self.k=None
        self.labels = None

    def compute(self, x:Multiset,k:int):
        self.x:Multiset = x
        self.k = k
        return self.kmeans()


    def initialize_centroids(self):
        """Randomly initialize k centroids from the dataset X."""
        _min,_max = self.x.fed_min(),self.x.fed_max()
        self.centroids = _min + (np.random.rand(self.k, len(_min)) * (_max - _min))

    def assign_clusters(self):
        """Assign each data point to the nearest centroid."""
        print('!!',self.centroids)
        distances = np.linalg.norm(self.x[:, np.newaxis] - self.centroids, axis=2)
        self.labels = np.argmin(distances, axis=1)

    def update_centroids(self):
        """Compute new centroids as the mean of all points assigned to each cluster."""
        new_centroids = np.array([self.x[self.labels == i].mean(axis=0) for i in range(self.k)])
        new_centroids=FedArray( new_centroids,self.x.client)
        new_centroids.fed_avg()
        return new_centroids

    def kmeans(self, max_iters=100, tol=1e-4):
        """Perform K-Means clustering."""
        self.initialize_centroids()
        for _ in range(max_iters):
            self.assign_clusters()
            new_centroids = self.update_centroids()
            if np.linalg.norm(new_centroids - self.centroids) < tol:
                break
            self.centroids = new_centroids
        return self.centroids
