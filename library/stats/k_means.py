import numpy as np
from typing import Tuple

from pythonProject.FedAvg6.system.client.aggregation_client import AggregationClient
from pythonProject.FedAvg6.library.templates.statistical_function import StatisticalFunction


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


import pandas as pd
import numpy as np
from typing import Tuple, Union

from pythonProject.FedAvg6.system.client.aggregation_client import AggregationClient
from pythonProject.FedAvg6.library.templates.statistical_function import StatisticalFunction

class KMeansPandas(StatisticalFunction):

    def __init__(self, client: AggregationClient):
        super().__init__(client)
        self.aggregator = self.get_pandas_aggregator()
        self.centroids: np.ndarray = np.array([])

    def assign_clusters(self, X: pd.DataFrame, centroids: np.ndarray) -> np.ndarray:
        """
        Assigns each data point to the nearest centroid.
        """
        dists = np.linalg.norm(X.values[:, np.newaxis] - centroids, axis=2)
        return np.argmin(dists, axis=1)

    def compute_local_sums_and_counts(self, X: pd.DataFrame, assignments: np.ndarray, k: int) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Computes the local sum of points and counts for each cluster.
        """
        n_features = X.shape[1]
        sums = np.zeros((k, n_features))
        counts = np.zeros(k)

        assignments_series = pd.Series(assignments, index=X.index)

        for i in range(k):
            mask = (assignments_series == i)
            if np.any(mask):
                sums[i] = X.loc[mask].sum(axis=0).values
                counts[i] = mask.sum()

        return sums, counts

    def initialize_centroids(self, X: pd.DataFrame, k: int) -> np.ndarray:
        """
        Samples local centroids and federates them to get a global initialization.
        """
        indices = np.random.choice(X.index, k, replace=False)
        print(f"Random indices are:\n {indices}")
        local_centroids = X.loc[indices].values
        print(f"Local centroids are:\n {local_centroids}")
        global_centroids = self.aggregator.fed_union(local_centroids)
        print(f"Global centroids are:\n {global_centroids}")
        return global_centroids

    def compute(self, x: pd.DataFrame, k: int):
        """
        Performs federated K-means clustering on the provided data.
        """
        max_iters: int = 10
        tol: float = 1e-4

        self.centroids = self.initialize_centroids(x, k)

        for _ in range(max_iters):
            assignments = self.assign_clusters(x, self.centroids)
            print(f"Assignments are:\n {assignments}")
            local_sums, local_counts = self.compute_local_sums_and_counts(x, assignments, k)
            print(f"Local sums and local counts are:\n {local_sums}\n{local_counts}")

            global_sums = self.aggregator.fed_sum(local_sums)
            global_counts = self.aggregator.fed_sum(local_counts)
            print(f"Global sums and global counts are:\n {global_sums}\n{global_counts}")

            # Compute new centroids based on global sums and counts
            new_centroids = np.zeros_like(self.centroids)
            for i in range(k):
                if global_counts[i] > 0:
                    new_centroids[i] = global_sums[i] / global_counts[i]
            
            print(f"New centroids are:\n {new_centroids}")

            # Check for convergence
            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids

            if shift < tol:
                print(f"K-means converged after {_ + 1} iterations.")
                break
        print("Final Centroids for this client:")
        print(self.centroids)

    def get_centroids(self) -> np.ndarray:
        return self.centroids


