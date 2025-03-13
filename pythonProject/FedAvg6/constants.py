from enum import Enum

class AGG(Enum):
    AVG = "AVG"
    SUM = "SUM"
    COUNT = "COUNT"
    def __str__(self):
        return self.name

