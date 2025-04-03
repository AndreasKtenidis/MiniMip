import numpy as np

from data.numpy_dataset.fed_multiset import Multiset
from function.abstract_function import AggFunc


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
        _min = self.x.fed_min()
        # _min,_max = self.x.fed_min(),self.x.fed_max()

        indices = np.random.choice(self.x.shape[0], self.k, replace=False)
        return self.x[indices]

    def assign_clusters(self):
        """Assign each data point to the nearest centroid."""
        distances = np.linalg.norm(self.x[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def update_centroids(self):
        """Compute new centroids as the mean of all points assigned to each cluster."""
        return np.array([self.x[self.labels == i].mean(axis=0) for i in range(self.k)])

    def kmeans(self, max_iters=100, tol=1e-4):
        """Perform K-Means clustering."""
        self.centroids = self.initialize_centroids()
        for _ in range(max_iters):
            self.labels = self.assign_clusters()
            new_centroids = self.update_centroids()
            if np.linalg.norm(new_centroids - self.centroids) < tol:
                break
            self.centroids = new_centroids
        return self.centroids
