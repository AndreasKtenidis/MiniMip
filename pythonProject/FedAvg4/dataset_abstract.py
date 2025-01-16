from abc import ABC, abstractmethod

class Dataset(ABC):
    @abstractmethod
    def local_sum(self, function_string, mapping):
        pass

    def local_count(self, function_string, mapping):
        pass