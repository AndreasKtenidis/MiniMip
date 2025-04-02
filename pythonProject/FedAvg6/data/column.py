from abc import ABC, abstractmethod
class Column(ABC):
    # def __init__(self, column_name, data_type):
    #     self.column_name = column_name
    #     self.data_type = data_type

    @abstractmethod
    def get_type(self):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def len(self):
        pass