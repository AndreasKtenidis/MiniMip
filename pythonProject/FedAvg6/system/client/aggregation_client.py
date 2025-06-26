import numpy as np
from abc import ABC,abstractmethod
from pandas import DataFrame

class AggregationClient(ABC):

    @abstractmethod
    def __global_sum__(self, local_sum):
        pass

    @abstractmethod
    def __global_min__(self, local_min):
        pass

    @abstractmethod
    def __global_max__(self, local_max):
        pass

    @abstractmethod
    def __global_union__(self, categories,c_type):
        pass

class NumpyAggClient( ABC):
    """A client for performing federated operations on numpy arrays.

    This class provides an interface for common federated aggregation operations
    that communicate with a central aggregation server through an AggregationClient.
    """

    def __init__(self,client:AggregationClient):
        """Initialize the NumpyAggClient with an aggregation client.

                Args:
                    client: An instance of AggregationClient that handles the actual
                           communication with the federated learning server.
                """
        self.client = client

    def fed_union(self,categories:np.array):
        """Compute the union of categories across all federated clients.

                Args:
                    categories: A numpy array of categories to be united with
                               categories from other clients.

                Returns:
                    A numpy array containing the union of all categories from
                    all clients, with duplicates removed.
                """
        _shape, _flattened,_type = NumpyAggClient._transform2(categories)
        _shape = (-1,) + _shape[1:]
        _ans= np.array(self.client.__global_union__(_flattened,_type))
        return NumpyAggClient._inv_transform(_shape, _ans)

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
        _shape, _flattened = NumpyAggClient._transform(array)
        _ans = self.client.__global_sum__(np.append(_flattened,1))
        tmp = np.array(_ans[:-1])/_ans[-1]
        return NumpyAggClient._inv_transform(_shape, tmp)

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
        _shape, _flattened = NumpyAggClient._transform(array)
        # Append weight for weighted averaging
        weighted_array = np.append(_flattened , 1)* weight
        _ans = self.client.__global_sum__(weighted_array)
        tmp = np.array(_ans[:-1]) / _ans[-1]  # weighted sum / total weight
        return NumpyAggClient._inv_transform(_shape, tmp)

    def fed_sum(self,array:np.ndarray):
        """Compute the federated sum of an array across all clients.

                Args:
                    array: The numpy array to be summed across all clients.

                Returns:
                    A numpy array with the same shape as input, containing the
                    element-wise sum across all clients' arrays.
                """
        _shape, _flattened = NumpyAggClient._transform(array)
        _ans = self.client.__global_sum__( _flattened)
        return NumpyAggClient._inv_transform(_shape, _ans)

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
        _shape, _flattened = NumpyAggClient._transform(np.sum(array, axis=0))
        _ans = self.client.__global_sum__(_flattened)
        return NumpyAggClient._inv_transform(_shape, _ans)

    def global_count(self,array:np.array):
        """Compute the federated count of samples across all clients.

                Args:
                    array: A numpy array whose first dimension represents samples.

                Returns:
                    The total count of samples across all clients.

                Note:
                    This effectively sums the first dimension sizes from all clients.
                """
        _ans = self.client.__global_sum__([array.shape[0]])
        return _ans[0]

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
        _shape, _flattened = NumpyAggClient._transform(np.sum(array, axis=0))
        _flattened = np.append(_flattened, array.shape[0])
        _ans = self.client.__global_sum__(_flattened)
        return NumpyAggClient._inv_transform(_shape, _ans[:-1]) / _ans[-1]

    def global_min(self,array:np.array):
        """Compute min along axis=0 and then federated min across clients.

                Args:
                    array: The numpy array to find minimum values from.

                Returns:
                    A numpy array with reduced dimensions (axis=0 removed),
                    containing the element-wise minimum across all clients.
                """
        _shape, _flattened = NumpyAggClient._transform(np.min(array, axis=0))
        _ans = self.client.__global_min__(_flattened)
        return NumpyAggClient._inv_transform(_shape, _ans)

    def global_max(self,array:np.array):
        """Compute max along axis=0 and then federated max across clients.

                Args:
                    array: The numpy array to find maximum values from.

                Returns:
                    A numpy array with reduced dimensions (axis=0 removed),
                    containing the element-wise maximum across all clients.
                """
        _shape, _flattened = NumpyAggClient._transform(np.max(array, axis=0))
        _ans = self.client.__global_max__(_flattened)
        return NumpyAggClient._inv_transform(_shape, _ans)

    @staticmethod
    def _transform(array:np.ndarray):
        try:
            out = array.flatten().astype(np.float64).tolist()
        except Exception as e:
            out = array.flatten().astype(str).tolist()
        return array.shape,out

    @staticmethod
    def _transform2(array: np.ndarray):
        try:
            out = array.flatten().astype(np.float64).tolist()
            return array.shape, out,np.float64
        except Exception as e:
            out = array.flatten().astype(str).tolist()
            return array.shape, out, str

    @staticmethod
    def _inv_transform(original_shape, answer):
        return np.asarray(answer).reshape(original_shape)

class PandasAggClient( ABC):

    def __init__(self,client:AggregationClient):
        self.client = client

    def global_sum(self,dataframe:DataFrame):
        _agg = DataFrame([dataframe.sum()])
        _shape, _flattened = PandasAggClient.transform(_agg)
        _ans = self.client.__global_sum__(_flattened)
        _ans = DataFrame(PandasAggClient.inv_transform(_shape, _ans))
        rename_mapping = {old: new for old, new in zip(_ans.columns, dataframe.columns)}
        _ans.rename(columns=rename_mapping, inplace=True)
        return _ans

    def global_count(self,dataframe:DataFrame):
        _agg = DataFrame([dataframe.count()])
        _shape, _flattened = PandasAggClient.transform(_agg)
        _ans = self.client.__global_sum__(_flattened)
        _ans = DataFrame(PandasAggClient.inv_transform(_shape, _ans))
        rename_mapping = {old: new for old, new in zip(_ans.columns, dataframe.columns)}
        _ans.rename(columns=rename_mapping, inplace=True)
        return _ans

    def global_min(self,dataframe:DataFrame):
        _agg = DataFrame([dataframe.min()])
        _shape, _flattened = PandasAggClient.transform(_agg)
        _ans = self.client.__global_min__(_flattened)
        _ans = DataFrame(PandasAggClient.inv_transform(_shape, _ans))
        rename_mapping = {old: new for old, new in zip(_ans.columns, dataframe.columns)}
        _ans.rename(columns=rename_mapping, inplace=True)
        return _ans

    def global_max(self,dataframe:DataFrame):
        _agg = DataFrame([dataframe.max()])
        _shape, _flattened = PandasAggClient.transform(_agg)
        _ans = self.client.__global_max__(_flattened)
        _ans = DataFrame(PandasAggClient.inv_transform(_shape, _ans))
        rename_mapping = {old: new for old, new in zip(_ans.columns, dataframe.columns)}
        _ans.rename(columns=rename_mapping, inplace=True)
        return _ans

    def global_avg(self,dataframe:DataFrame):
        _agg = DataFrame([dataframe.mean()])
        _shape, _flattened = PandasAggClient.transform(_agg)
        _flattened= np.append(_flattened, dataframe.count())
        _ans = self.client.__global_sum__(_flattened)
        _ans =np.array(_ans[0:len(_ans)//2]) / _ans[len(_ans)//2:]
        _ans = DataFrame(PandasAggClient.inv_transform(_shape,_ans))
        rename_mapping = {old: new for old, new in zip(_ans.columns, dataframe.columns)}
        _ans.rename(columns=rename_mapping, inplace=True)
        return _ans

    @staticmethod
    def transform(array):
        out = array.to_numpy().flatten().astype(np.float64).tolist()
        return array.shape, out

    @staticmethod
    def inv_transform(original_shape, answer):
        return np.asarray(answer).reshape(original_shape)