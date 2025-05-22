import numpy as np


class MyStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None  # Standard deviation (σ)

    def fit(self, X):
        """Compute mean and standard deviation for scaling."""
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        """Scale data to have mean=0 and std=1."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Fit the scaler first!")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


# Example Usage
if __name__ == "__main__":
    data = np.array([[1, 2], [3, 4], [5, 6]])

    # Initialize and fit scaler
    scaler = MyStandardScaler()
    scaler.fit(data)

    # Transform data
    scaled_data = scaler.transform(data)

    print("Original Data:\n", data)
    print("Scaled Data (Mean=0, Std=1):\n", scaled_data)
    print("Mean:", scaler.mean_)
    print("Std Dev:", scaler.scale_)