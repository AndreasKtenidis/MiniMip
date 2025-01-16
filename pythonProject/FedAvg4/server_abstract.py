from enum import Enum
from abc import ABC, abstractmethod

class AGG(Enum):
    AVG = "AVG"
    SUM = "SUM"
    COUNT = "COUNT"
    def __str__(self):
        return self.name

class PARAMS(Enum):
    AGG_FUNC = "AGG_FUNC"
    MAPPING = "MAPPING"
    COL_FUNC = "COL_FUNC"
    RESULTS = "RESULTS"

    def __str__(self):
        return self.name


class Aggregation:
    def __init__(self, agg_func:AGG, col_func:str, mapping:dict[str,str]):
        self.agg_func = agg_func
        self.col_func = col_func
        self.mapping = mapping

    # Getter for agg_func
    def get_agg_func(self) -> AGG:
        return self.agg_func

    # Getter for col_func
    def get_col_func(self) -> str:
        return self.col_func

    # Getter for mapping
    def get_mapping(self) -> dict[str, str]:
        return self.mapping

class Message(ABC):

    meter = 0

    def __init__(self, dataset: str):
        self.id =Message.meter
        Message.meter += 1
        print(self.id)


class Assignments:
    def __init__(self, dataset:str):
        self.dataset = dataset
        self.aggregations = {}

    def add_aggregation(self, agg_operation:Aggregation):
        self.aggregations.append(agg_operation)

    @abstractmethod
    def get_message(self):
        pass

class ServerAbstract:
    def __init__(self, dataset:str):
        self.dataset = dataset
        self.aggregations = Assignments(dataset)

