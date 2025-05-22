from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class IFederatedAggregations(ABC):
    """Interface defining federated aggregation operations for numpy arrays."""

    @abstractmethod
    def fed_union(self, categories: np.ndarray) -> np.ndarray:
        """Compute set union of categories across all clients.

        Args:
            categories: Array of categorical values from this client

        Returns:
            Array containing the unified categories from all clients
        """
        pass

    @abstractmethod
    def fed_avg(self, array: np.ndarray) -> np.ndarray:
        """Compute element-wise average across all clients' arrays.

        Args:
            array: Local array to contribute to the average

        Returns:
            Array with same shape as input containing global average
        """
        pass

    @abstractmethod
    def fed_weighted_avg(self, array: np.ndarray, weight: float) -> np.ndarray:
        """Compute weighted average across all clients.

        Args:
            array: Local array to contribute
            weight: This client's weight (typically sample count)

        Returns:
            Array with same shape as input: sum(weights*arrays)/sum(weights)
        """
        pass

    @abstractmethod
    def fed_sum(self, array: np.ndarray) -> np.ndarray:
        """Compute element-wise sum across all clients' arrays.

        Args:
            array: Local array to contribute

        Returns:
            Array with same shape as input containing global sum
        """
        pass

    @abstractmethod
    def global_sum(self, array: np.ndarray) -> np.ndarray:
        """Sum array along axis=0, then federated sum across clients.

        Args:
            array: Array to sum (will be summed along axis=0 first)

        Returns:
            Array with reduced dimension containing federated sum
        """
        pass

    @abstractmethod
    def global_count(self, array: np.ndarray) -> int:
        """Count total samples across all clients.

        Args:
            array: Array where shape[0] represents local sample count

        Returns:
            Total count of samples across federation
        """
        pass

    @abstractmethod
    def global_avg(self, array: np.ndarray) -> np.ndarray:
        """Sum array along axis=0, then compute federated average.

        Args:
            array: Array to process (summed along axis=0 first)

        Returns:
            Array with reduced dimension: total_sum/total_samples
        """
        pass

    @abstractmethod
    def global_min(self, array: np.ndarray) -> np.ndarray:
        """Find minimum along axis=0, then federated minimum across clients.

        Args:
            array: Array to process

        Returns:
            Array with reduced dimension containing global minimums
        """
        pass

    @abstractmethod
    def global_max(self, array: np.ndarray) -> np.ndarray:
        """Find maximum along axis=0, then federated maximum across clients.

        Args:
            array: Array to process

        Returns:
            Array with reduced dimension containing global maximums
        """
        pass

    @staticmethod
    @abstractmethod
    def transform(array: np.ndarray) -> Tuple[Tuple, np.ndarray]:
        """Flatten array while preserving shape information.

        Args:
            array: Array to flatten

        Returns:
            Tuple of (original_shape, flattened_array)
        """
        pass

    @staticmethod
    @abstractmethod
    def inv_transform(shape: Tuple, array: np.ndarray) -> np.ndarray:
        """Reconstruct array to original shape.

        Args:
            shape: Original array shape
            array: Flattened array

        Returns:
            Array reshaped to original dimensions
        """
        pass