import numpy as np
from abc import ABC,abstractmethod


class NumpyAggregationClient(ABC):

    @abstractmethod
    def __global_sum__(self, local_sum):
        pass

    @abstractmethod
    def __global_min__(self, local_min):
        pass

    @abstractmethod
    def __global_max__(self, local_max):
        pass

    def fed_sum(self, array_sum:np.ndarray):
        _shape,_flattened=self.transform(array_sum)
        _ans = self.__global_sum__(_flattened)
        return self.inv_transform(_shape,_ans)

    def fed_count(self,array_count:int):
        _ans = self.__global_sum__([array_count])
        return _ans[0]

    def fed_avg(self,array_sum:np.ndarray,array_count:int):
        _shape, _flattened = self.transform(array_sum)
        _flattened = np.append(_flattened, array_count)
        _ans = self.__global_sum__(_flattened)
        return self.inv_transform(_shape,_ans[:-1])/_ans[-1]

    def fed_min(self,array_min:np.ndarray):
        _shape, _flattened = self.transform(array_min)
        _ans = self.__global_min__(_flattened)
        return self.inv_transform(_shape,_ans)

    def fed_max(self,array_max:np.ndarray):
        _shape, _flattened = self.transform(array_max)
        _ans = self.__global_max__(_flattened)
        return self.inv_transform(_shape,_ans)

    @staticmethod
    def transform(array:np.ndarray):
        out = array.flatten().astype(np.float64).tolist()

        return array.shape,out


    @staticmethod
    def inv_transform(original_shape,answer):
        return np.asarray(answer).reshape(original_shape)