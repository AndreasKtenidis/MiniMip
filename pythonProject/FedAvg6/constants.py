from enum import Enum

class AGG(Enum):
    AVG = "AVG"
    SUM = "SUM"
    COUNT = "COUNT"
    MIN = "MIN"
    MAX = "MAX"
    def __str__(self):
        return self.name

client_count:int=3