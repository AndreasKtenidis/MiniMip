import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs  # Only used for generating synthetic data
from scipy.spatial.distance import cdist

from library.templates.statistical_function import StatisticalFunction

# # Step 1: Load synthetic blob dataset
# points, _ = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=1.0, random_state=38)
# points = pd.DataFrame(points, columns=['x', 'y'])
#
# k = 3

class KMeansPandas(StatisticalFunction):

    def compute(self, points, k):
        aggregator = self.get_numpy_aggregator()
        centroids: pd.DataFrame = points.sample(n=k, random_state=1).reset_index(drop=True)

        # Step 3: Pure Pandas k-means loop
        for iteration in range(10):
            points['cluster'] = points.apply(
                lambda row: KMeansPandas.find_closest_centroid(row, centroids),
                axis=1
            )
            centroid_sum_max = points.groupby('cluster').agg(['sum', 'count'])
            centroid_sum_max.iloc[:, :]  =aggregator.fed_sum(centroid_sum_max.values)
            centroids = self.recompute_centroids(centroid_sum_max)
        return centroids


    @staticmethod
    def find_closest_centroid(point_row, centroids):
        point = point_row.values[0:centroids.shape[1]].reshape(1, -1)
        distances = cdist(point, centroids.values, metric='euclidean')
        closest_idx = np.argmin(distances)
        return centroids.index[closest_idx]

    @staticmethod
    def recompute_centroids(summary: pd.DataFrame):
        cols = list(summary.columns)
        n_rows = len(summary)
        n_features = len(cols) // 2  # assuming pairs of columns (_sum and _count)

        # Initialize output array: shape (n_rows, n_features)
        out = np.zeros((n_rows, n_features))

        for i in range(0, len(cols) - 1, 2):
            feature_idx = i // 2
            sum_col = cols[i]
            count_col = cols[i + 1]
            out[:, feature_idx] = summary[sum_col].values / summary[count_col].values
        return pd.DataFrame(out)

