import numpy as np
import pandas as pd

from data.pandas_federation.fed_table import FedDataFrame
from function.abstract_function import AggFunc


class KMeans(AggFunc):

    def __init__(self):
        self.x = None
        self.centroids = None
        self.k=None
        self.labels = None
        self.agg_cols = None

    def compute(self, x:FedDataFrame,k:int):
        self.x:FedDataFrame = x
        self.k = k
        self.agg_cols = self.x.columns
        return self.kmeans()

    def initialize_centroids(self):
        """Randomly initialize k centroids from the dataset X."""
        _min,_max = self.x.fed_min(),self.x.fed_max()
        _multi_min = pd.concat([_min] * self.k, ignore_index=True)
        self.centroids = _multi_min + np.random.rand(self.k, _min.shape[1])*(pd.concat([_max] * self.k, ignore_index=True)-_multi_min)
        self.centroids .index.name = 'centroid'


    @staticmethod
    def assign_clusters(points, centroids):
        diff = points - centroids
        distances = np.linalg.norm(diff,axis=1)
        return np.argmin(distances)


    def update_centroids(self):
        """Compute new centroids as the mean of all points assigned to each cluster."""

        tmp = self.x.groupby(['centroid'])[self.agg_cols].agg(['mean', 'count']).sort_index()
        for i in range(0, self.k):
            point_i=tmp.loc[[0]].xs('mean', axis=1, level=1)
            print(point_i)



        # new_centroids = new_centroids.xs('mean', axis=1, level=1)
        # full_index = pd.RangeIndex(0, self.k)
        # new_centroids = FedDataFrame(new_centroids.reindex(full_index),self.x.client)


        return None

    def kmeans(self, max_iters=100, tol=1e-4):
        """Perform K-Means clustering."""
        self.initialize_centroids()
        for _ in range(max_iters):
            # Assigns each point to a centroid
            self.x['centroid'] = self.x.apply(self.assign_clusters, axis=1, args=(self.centroids,))
            new_centroids = self.update_centroids()

            if np.linalg.norm(new_centroids - self.centroids) < tol:
                break
            self.centroids = new_centroids
        return self.centroids
