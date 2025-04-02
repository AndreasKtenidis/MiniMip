from abc import ABC, abstractmethod
class Column(ABC):

    @abstractmethod
    def count(self):
        pass

    @abstractmethod
    def sum(self):
        pass

    @abstractmethod
    def avg(self):
        pass

    @abstractmethod
    def min(self):
        pass

    @abstractmethod
    def max(self):
        pass