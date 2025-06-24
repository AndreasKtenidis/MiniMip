import numpy as np
from typing import Tuple

from system.client.aggregation_client import AggregationClient
from library.templates.statistical_function import StatisticalFunction

class KMeans(StatisticalFunction):
    def __init__(self, client: AggregationClient):
        super().__init__(client)
        self.aggregator = self.get_numpy_aggregator()

    def assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest centroid."""
        dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # Shape: (n_samples, k)
        return np.argmin(dists, axis=1)

    def compute_local_sums_and_counts(self, X: np.ndarray, assignments: np.ndarray, k: int) -> Tuple[
        np.ndarray, np.ndarray]:
        """Compute local sum of points and counts for each cluster."""
        n_features = X.shape[1]
        sums = np.zeros((k, n_features))
        counts = np.zeros(k)

        for i in range(k):
            mask = (assignments == i)
            if np.any(mask):
                sums[i] = X[mask].sum(axis=0)
                counts[i] = mask.sum()

        return sums, counts

    def initialize_centroids(self, X: np.ndarray, k: int) -> np.ndarray:
        """Sample local centroids and federate a global initialization."""
        indices = np.random.choice(X.shape[0], k, replace=False)
        local_centroids = X[indices]
        global_centroids = self.aggregator.fed_union(local_centroids)
        return global_centroids

    def compute(self, x: np.ndarray, k: int):

        """Perform federated K-means clustering."""
        max_iters: int = 10
        tol: float = 1e-4
        self.centroids = self.initialize_centroids(x, k)

        for _ in range(max_iters):
            assignments = self.assign_clusters(x, self.centroids)
            local_sums, local_counts = self.compute_local_sums_and_counts(x, assignments, k)

            global_sums = self.aggregator.fed_sum(local_sums)
            global_counts = self.aggregator.fed_sum(local_counts)

            new_centroids = np.zeros_like(self.centroids)
            for i in range(k):
                if global_counts[i] > 0:
                    new_centroids[i] = global_sums[i] / global_counts[i]

            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids

            if shift < tol:
                break
        print(self.centroids)

    def get_centroids(self) -> np.ndarray:
        return self.centroids
