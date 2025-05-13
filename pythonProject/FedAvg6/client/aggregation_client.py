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
    def __global_union__(self, categories):
        pass

class NumpyAggClient( ABC):

    def __init__(self,client:AggregationClient):
        self.client = client

    def fed_avg(self,array:np.ndarray):
        _shape, _flattened = NumpyAggClient.transform(array)
        _ans = self.client.__global_sum__(np.append(_flattened,1))
        tmp = np.array(_ans[:-1])/_ans[-1]
        return NumpyAggClient.inv_transform(_shape, tmp)

    def fed_sum(self,array:np.ndarray):
        _shape, _flattened = NumpyAggClient.transform(array)
        _ans = self.client.__global_sum__( _flattened)
        return NumpyAggClient.inv_transform(_shape, _ans)

    def global_sum(self,array:np.array):
        _shape, _flattened = NumpyAggClient.transform(np.sum(array, axis=0))
        _ans = self.client.__global_sum__(_flattened)
        return NumpyAggClient.inv_transform(_shape, _ans)

    def global_count(self,array:np.array):
        _ans = self.client.__global_sum__([array.shape[0]])
        return _ans[0]

    def global_avg(self,array:np.array):
        _shape, _flattened = NumpyAggClient.transform(np.sum(array, axis=0))
        _flattened = np.append(_flattened, array.shape[0])
        _ans = self.client.__global_sum__(_flattened)
        return NumpyAggClient.inv_transform(_shape, _ans[:-1]) / _ans[-1]

    def global_min(self,array:np.array):
        _shape, _flattened = NumpyAggClient.transform(np.min(array, axis=0))
        _ans = self.client.__global_min__(_flattened)
        return NumpyAggClient.inv_transform(_shape, _ans)

    def global_max(self,array:np.array):
        _shape, _flattened = NumpyAggClient.transform(np.max(array, axis=0))
        _ans = self.client.__global_max__(_flattened)
        return NumpyAggClient.inv_transform(_shape, _ans)

    @staticmethod
    def transform(array:np.ndarray):
        out = array.flatten().astype(np.float64).tolist()
        return array.shape,out

    @staticmethod
    def inv_transform(original_shape,answer):
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
        _ans = self.client.__global_sum__(_flattened)
        _ans = DataFrame(PandasAggClient.inv_transform(_shape, _ans))
        rename_mapping = {old: new for old, new in zip(_ans.columns, dataframe.columns)}
        _ans.rename(columns=rename_mapping, inplace=True)
        return _ans

    def global_max(self,dataframe:DataFrame):
        _agg = DataFrame([dataframe.max()])
        _shape, _flattened = PandasAggClient.transform(_agg)
        _ans = self.client.__global_sum__(_flattened)
        _ans = DataFrame(PandasAggClient.inv_transform(_shape, _ans))
        rename_mapping = {old: new for old, new in zip(_ans.columns, dataframe.columns)}
        _ans.rename(columns=rename_mapping, inplace=True)
        return _ans

    def global_avg(self,dataframe:DataFrame):
        _shape, _flattened = PandasAggClient.transform(np.sum(dataframe, axis=0))
        _flattened = np.append(_flattened, dataframe.shape[0])
        _ans = self.client.__global_sum__(_flattened)
        _ans = DataFrame(PandasAggClient.inv_transform(_shape, _ans[:-1]) / _ans[-1])
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