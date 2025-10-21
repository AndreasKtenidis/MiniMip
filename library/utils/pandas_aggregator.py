from abc import ABC
from typing import Union

import numpy as np
import pandas as pd
from pandas import DataFrame

from library.utils.aggregation_client import AggregationClientInterface


class PandasAggClient( ):

    def __init__(self, client:AggregationClientInterface):
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
        _agg = DataFrame([dataframe.sum()])
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
