import numpy as np
from abc import ABC,abstractmethod
from pandas import DataFrame

class AggregationClient(ABC):

    @abstractmethod
    def fed_union(self, categories: np.array):
        """Compute the union of categories across all federated clients.

                Args:
                    categories: A numpy array of categories to be united with
                               categories from other clients.

                Returns:
                    A numpy array containing the union of all categories from
                    all clients, with duplicates removed.
                """

    @abstractmethod
    def fed_avg(self, array: np.ndarray):
        """Compute the federated average of an array across all clients.

                Args:
                    array: The numpy array to be averaged across all clients.

                Returns:
                    A numpy array with the same shape as input, containing the
                    element-wise average across all clients' arrays.

                Note:
                    Internally flattens the array for transmission and appends
                    a count (1) for proper averaging.
                """

    @abstractmethod
    def fed_weighted_avg(self, array: np.ndarray, weight: float) -> np.ndarray:
        """Compute federated weighted average of an array across all clients.

        Args:
            array: The numpy array to be averaged across all clients.
            weight: The weight (typically sample count) for this client's array.
                   Weights from all clients will be summed for normalization.

        Returns:
            A numpy array with the same shape as input, containing the
            element-wise weighted average across all clients' arrays.

        Note:
            - Follows the formula: sum(weight_i * array_i) / sum(weights)
            - Internally flattens the array for transmission
            - The weight should typically be positive
            - If all weights are 1, this is equivalent to fed_avg()
        """

    @abstractmethod
    def fed_sum(self, array: np.ndarray):
        """Compute the federated sum of an array across all clients.

                Args:
                    array: The numpy array to be summed across all clients.

                Returns:
                    A numpy array with the same shape as input, containing the
                    element-wise sum across all clients' arrays.
                """

    @abstractmethod
    def global_sum(self, array: np.array):
        """Compute sum along axis=0 and then federated sum across clients.

                Args:
                    array: The numpy array to be summed (along axis=0 first).

                Returns:
                    A numpy array with reduced dimensions (axis=0 removed),
                    containing the federated sum of all clients' sums.

                Note:
                    This is different from fed_sum as it first reduces the array
                    by summing along axis=0 before federated aggregation.
                """

    @abstractmethod
    def global_count(self, array: np.array):
        """Compute the federated count of samples across all clients.

                Args:
                    array: A numpy array whose first dimension represents samples.

                Returns:
                    The total count of samples across all clients.

                Note:
                    This effectively sums the first dimension sizes from all clients.
                """

    @abstractmethod
    def global_avg(self, array: np.array):
        """Compute federated average of array sums across clients.

                Args:
                    array: The numpy array to be processed (summed along axis=0 first).

                Returns:
                    A numpy array with reduced dimensions (axis=0 removed),
                    containing the federated average across all clients.

                Note:
                    Similar to global_sum but divides by total sample count.
                    More efficient than fed_avg for large arrays as it reduces first.
                """

    @abstractmethod
    def global_min(self, array: np.array):
        """Compute min along axis=0 and then federated min across clients.

                Args:
                    array: The numpy array to find minimum values from.

                Returns:
                    A numpy array with reduced dimensions (axis=0 removed),
                    containing the element-wise minimum across all clients.
                """

    @abstractmethod
    def global_max(self, array: np.array):
        """Compute max along axis=0 and then federated max across clients.

                Args:
                    array: The numpy array to find maximum values from.

                Returns:
                    A numpy array with reduced dimensions (axis=0 removed),
                    containing the element-wise maximum across all clients.
                """

class NumpyAggClient( ABC):
    """A client for performing federated operations on numpy arrays.

    This class provides an interface for common federated aggregation operations
    that communicate with a central aggregation server through an AggregationClient.
    """

    @abstractmethod
    def fed_union(self,categories:np.array):
        """Compute the union of categories across all federated clients.

                Args:
                    categories: A numpy array of categories to be united with
                               categories from other clients.

                Returns:
                    A numpy array containing the union of all categories from
                    all clients, with duplicates removed.
                """

    @abstractmethod
    def fed_avg(self,array:np.ndarray):
        """Compute the federated average of an array across all clients.

                Args:
                    array: The numpy array to be averaged across all clients.

                Returns:
                    A numpy array with the same shape as input, containing the
                    element-wise average across all clients' arrays.

                Note:
                    Internally flattens the array for transmission and appends
                    a count (1) for proper averaging.
                """

    @abstractmethod
    def fed_weighted_avg(self, array: np.ndarray, weight: float) -> np.ndarray:
        """Compute federated weighted average of an array across all clients.

        Args:
            array: The numpy array to be averaged across all clients.
            weight: The weight (typically sample count) for this client's array.
                   Weights from all clients will be summed for normalization.

        Returns:
            A numpy array with the same shape as input, containing the
            element-wise weighted average across all clients' arrays.

        Note:
            - Follows the formula: sum(weight_i * array_i) / sum(weights)
            - Internally flattens the array for transmission
            - The weight should typically be positive
            - If all weights are 1, this is equivalent to fed_avg()
        """

    @abstractmethod
    def fed_sum(self,array:np.ndarray):
        """Compute the federated sum of an array across all clients.

                Args:
                    array: The numpy array to be summed across all clients.

                Returns:
                    A numpy array with the same shape as input, containing the
                    element-wise sum across all clients' arrays.
                """

    @abstractmethod
    def global_sum(self,array:np.array):
        """Compute sum along axis=0 and then federated sum across clients.

                Args:
                    array: The numpy array to be summed (along axis=0 first).

                Returns:
                    A numpy array with reduced dimensions (axis=0 removed),
                    containing the federated sum of all clients' sums.

                Note:
                    This is different from fed_sum as it first reduces the array
                    by summing along axis=0 before federated aggregation.
                """

    @abstractmethod
    def global_count(self,array:np.array):
        """Compute the federated count of samples across all clients.

                Args:
                    array: A numpy array whose first dimension represents samples.

                Returns:
                    The total count of samples across all clients.

                Note:
                    This effectively sums the first dimension sizes from all clients.
                """

    @abstractmethod
    def global_avg(self,array:np.array):
        """Compute federated average of array sums across clients.

                Args:
                    array: The numpy array to be processed (summed along axis=0 first).

                Returns:
                    A numpy array with reduced dimensions (axis=0 removed),
                    containing the federated average across all clients.

                Note:
                    Similar to global_sum but divides by total sample count.
                    More efficient than fed_avg for large arrays as it reduces first.
                """

    @abstractmethod
    def global_min(self,array:np.array):
        """Compute min along axis=0 and then federated min across clients.

                Args:
                    array: The numpy array to find minimum values from.

                Returns:
                    A numpy array with reduced dimensions (axis=0 removed),
                    containing the element-wise minimum across all clients.
                """

    @abstractmethod
    def global_max(self,array:np.array):
        """Compute max along axis=0 and then federated max across clients.

                Args:
                    array: The numpy array to find maximum values from.

                Returns:
                    A numpy array with reduced dimensions (axis=0 removed),
                    containing the element-wise maximum across all clients.
                """