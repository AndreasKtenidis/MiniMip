from abc import ABC,abstractmethod
from typing import Dict, List,Optional
from numbers import Number

class ClientInterface(ABC):

    def get_result(self, agg_funcs: List[str], function, dataset: str, mapping: Dict[str:str])->Dict[str, List[Number]]:
        answer={}
        for agg_func in agg_funcs:
            if agg_func=='sum':
                answer['sum']=self.local_sum(function,dataset,mapping)
            elif agg_func=='count':
                answer['count']=self.local_count(dataset,mapping)
            else:
                raise ValueError("The aggregation function ",agg_func," is not accepted")

    @abstractmethod
    def local_sum(self, function, dataset, mapping):
        pass

    @abstractmethod
    def local_count(self, dataset, mapping):
        pass
