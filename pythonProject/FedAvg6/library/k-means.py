import numpy as np


def initialize_centroids(X, k):
    """Randomly initialize k centroids from the dataset X."""
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]


def assign_clusters(X, centroids):
    """Assign each data point to the nearest centroid."""
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, k):
    """Compute new centroids as the mean of all points assigned to each cluster."""
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])


def kmeans(X, k, max_iters=100, tol=1e-4):
    """Perform K-Means clustering."""
    centroids = initialize_centroids(X, k)

    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)

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
