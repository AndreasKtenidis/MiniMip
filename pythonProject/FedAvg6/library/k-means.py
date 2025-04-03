import numpy as np

from data.numpy_dataset.multiset import Multiset
from function.abstract_function import AggFunc


class KMeans(AggFunc):

    def __init__(self):
        self.x = None
        self.centroids = None
        self.k=None
        self.labels = None

    def compute(self, x:Multiset,k:int):
        self.x = x
        self.k = k

    def initialize_centroids(self):
        """Randomly initialize k centroids from the dataset X."""
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
        centroids = self.initialize_centroids()
        labels=None
        for _ in range(max_iters):
            labels = self.assign_clusters()
            new_centroids = self.update_centroids()
            if np.linalg.norm(new_centroids - centroids) < tol:
                break
            centroids = new_centroids
        return labels, centroids


    # Example usage
    if __name__ == "__main__":
        from sklearn.datasets import make_blobs
        import matplotlib.pyplot as plt

        # Generate sample data
        X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

        # Run K-Means
        labels, centroids = kmeans(X, 3)

        # Plot results
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
        plt.legend()
        plt.show()
