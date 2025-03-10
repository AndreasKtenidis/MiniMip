from enum import Enum

class AGG(Enum):
    AVG = "AVG"
    SUM = "SUM"
    COUNT = "COUNT"
    def __str__(self):
        return self.name


class Database(Enum):
    OPP = 'operation_id'
    CLIENT = 'client_id'
    ROUND = 'round'
    AGG = 'agg_func'
    VALUE = 'value'
    CLIENT_COUNT = 'client_count'



class PARAMS(Enum):
    FUNCTION = "FUNCTION"
    DATASET = "DATASET"
    OPERATION_ID = "OPERATION_ID"
    AGG_FUNC = "AGG_FUNC"
    MAPPING = "MAPPING"
    COL_FUNC = "COL_FUNC"
    RESULTS = "RESULTS"

    def __str__(self):
        return self.name
