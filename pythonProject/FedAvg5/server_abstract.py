from enum import Enum

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
