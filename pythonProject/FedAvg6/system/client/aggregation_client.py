import numpy as np
from abc import ABC,abstractmethod
import pandas as pd
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

from typing import Union, Tuple

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

    def fed_union(self, categories: Union[pd.Series, pd.DataFrame, np.ndarray]):
        """
        Compute the union of categories across all federated clients.
        """
        _original_type, _original_shape, _original_columns, _original_index, _flattened, _dtype = \
            PandasAggClient._transform_for_union(categories)

        print(f"Info: _original_type --> {_original_type}, _original_shape --> {_original_shape}, _original_columns --> {_original_columns}, _original_index --> {_original_index}, _flattened --> {_flattened}, _dtype --> {_dtype}")
        # Determine c_type for GRPCClient based on inferred dtype
        if np.issubdtype(_dtype, np.integer):
            _c_type = np.int64
        elif np.issubdtype(_dtype, np.floating):
            _c_type = np.float64
        else:
            _c_type = str

        _ans = self.client.__global_union__(_flattened, _c_type)
        print(f"Global union:\n {_ans}")
        return PandasAggClient._inv_transform_from_union(_original_type, _original_shape, _original_columns, _original_index, _ans, _dtype)


    def fed_sum(self, data: Union[pd.Series, pd.DataFrame, np.ndarray]) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
        """
        Compute the federated sum of data across all clients.
        """
        _original_type, _original_shape, _original_columns, _original_index, _flattened, _dtype = \
            PandasAggClient._transform(data)

        print(f"Info: _original_type --> {_original_type}, _original_shape --> {_original_shape}, _original_columns --> {_original_columns}, _original_index --> {_original_index}, _flattened --> {_flattened}, _dtype --> {_dtype}")
        _ans = self.client.__global_sum__(_flattened)
        print(f"Global sum:\n {_ans}")
        return PandasAggClient._inv_transform(_original_type, _original_shape, _original_columns, _original_index, _ans, _dtype)

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

    @staticmethod
    def _transform(data: Union[pd.Series, pd.DataFrame, np.ndarray]) -> Tuple[str, Tuple, Union[list, str], list, list, np.dtype]:
        """
        Transforms a pandas Series/DataFrame or numpy array into a flattened list for gRPC transmission,
        and returns metadata for reconstruction. Used primarily for sum/min/max, expecting numerical data.
        """
        if isinstance(data, pd.Series):
            return 'series', data.shape, data.name, data.index.tolist(), data.astype(float).tolist(), data.dtype
        elif isinstance(data, pd.DataFrame):
            return 'dataframe', data.shape, data.columns.tolist(), data.index.tolist(), data.values.astype(float).flatten().tolist(), data.values.dtype
        elif isinstance(data, np.ndarray):
            return 'ndarray', data.shape, None, None, data.astype(float).flatten().tolist(), data.dtype
        else:
            raise TypeError(f"Unsupported data type for transformation: {type(data)}. Must be pandas Series, DataFrame, or numpy array.")

    @staticmethod
    def _transform_for_union(data: Union[pd.Series, pd.DataFrame, np.ndarray]):
        """
        Transforms data for union operations, which can handle mixed types (numerical or categorical).
        It determines the appropriate c_type for GRPCClient based on the data's dtype.
        """
        if isinstance(data, pd.Series):
            return 'series', data.shape, data.name, data.index.tolist(), data.tolist(), data.dtype
        elif isinstance(data, pd.DataFrame):
            return 'dataframe', data.shape, data.columns.tolist(), data.index.tolist(), data.values.flatten().tolist(), data.values.dtype
        elif isinstance(data, np.ndarray):
            return 'ndarray', data.shape, None, None, data.flatten().tolist(), data.dtype
        else:
            raise TypeError(f"Unsupported data type for transformation for union: {type(data)}. Must be pandas Series, DataFrame, or numpy array.")


    @staticmethod
    def _inv_transform(original_type: str, original_shape: Tuple, original_columns: Union[list, str], original_index: list, answer: list, dtype: np.dtype) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
        """
        Inverse transforms a list received from gRPC back into the original pandas Series/DataFrame or numpy array.
        Used for sum/min/max.
        """
        if original_type == 'series':
            return pd.Series(answer, index=original_index, name=original_columns, dtype=dtype)
        elif original_type == 'dataframe':
            try:
                reshaped_array = np.asarray(answer, dtype=dtype).reshape(original_shape)
                return pd.DataFrame(reshaped_array, columns=original_columns, index=original_index)
            except ValueError:
                print(f"Warning: Could not reshape array to original DataFrame shape {original_shape}. Returning Series.")
                return pd.Series(answer, name='aggregated_data', dtype=dtype)
        elif original_type == 'ndarray':
            try:
                return np.asarray(answer, dtype=dtype).reshape(original_shape)
            except ValueError:
                print(f"Warning: Could not reshape array to original NumPy shape {original_shape}. Returning 1D array.")
                return np.asarray(answer, dtype=dtype)
        return answer

    @staticmethod
    def _inv_transform_from_union(original_type: str, original_shape: Tuple, original_columns: Union[list, str], original_index: list, answer: list, dtype: np.dtype) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
        """
        Inverse transforms a list from gRPC back into the original pandas Series/DataFrame or numpy array.
        Used for union operations.
        """
        if original_type == 'series':
            return pd.Series(answer, name='union_categories', dtype=dtype)
        elif original_type == 'dataframe':
            return pd.Series(answer, name='union_items', dtype=dtype)
        elif original_type == 'ndarray':
            # This is primarily for centroids. If the union operation for centroids results
            # in a 1D list, try to reshape it back to a 2D array, assuming the original
            # number of features is preserved.
            if original_shape and original_shape[1] > 0 and len(answer) % original_shape[1] == 0:
                try:
                    return np.asarray(answer, dtype=dtype).reshape(-1, original_shape[1])
                except ValueError:
                    print(f"Warning: Could not reshape union result to original centroid shape {original_shape}. Returning 1D array.")
                    return np.asarray(answer, dtype=dtype)
            return np.asarray(answer, dtype=dtype)
        return answer

class GrizzlyAggClient( ABC):
    """A client for performing federated operations on Grizzly DataFrames.

    This class provides an interface for common federated aggregation operations
    that communicate with a central aggregation server through an AggregationClient.
    """

    def __init__(self,client:AggregationClient):
        self.client = client
    
    def global_sum(self, dataframe):
        _agg = dataframe.sum()
        _ans = self.client.__global_sum__([_agg])
        return _ans
    
    def global_avg(self, dataframe):
        _agg = dataframe.mean()
        means = [row[1] for row in _agg.collect()]
        counts_df = dataframe.count()
        counts = [row[1] for row in counts_df.collect()]
        means_with_count = means + counts
        _ans = self.client.__global_sum__(means_with_count)
        _ans = np.array(_ans[0:len(_ans)//2]) / _ans[len(_ans)//2:]
        return _ans

    def global_count(self, dataframe):
        local_count = dataframe.count()
        total_count = self.client.__global_sum__([local_count])
        return total_count[0]
