from abc import ABC,abstractmethod

class LocalStorage(ABC):

    @abstractmethod
    def store(self, key:str, value):
        pass


    @abstractmethod
    def load(self,x:str):
        pass

class NumpyAggregatorServer(ABC):

    @abstractmethod
    def __global_sum__(self, local_sum)->float:
        pass

    @abstractmethod
    def __global_count__(self, local_count)->int:
        pass

    @abstractmethod
    def __global_avg__(self, local_sum, local_count) -> int:
        pass
