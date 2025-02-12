from abc import ABC,abstractmethod

class LocalStorage(ABC):

    @abstractmethod
    def store(self, key:str, value):
        pass


    @abstractmethod
    def load(self,x:str):
        pass

