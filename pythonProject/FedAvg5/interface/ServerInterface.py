from abc import ABC, abstractmethod
from typing import Dict, List,Optional
from numbers import Number


class ClientInteface(ABC):

    def agg_sum(self,metadata,  function, dataset: str,  mapping: Dict[str,str]) -> Number:
        results = self.get_results(metadata,["sum"], function, dataset, mapping)
        _sum = 0
        for result in results["sum"]:
            _sum += result
        return _sum


    def agg_avg(self,metadata, function, dataset: str, mapping: Dict[str,str]) -> Optional[Number]:
        results = self.get_results(metadata,["sum", "count"], function, dataset, mapping)
        _sum = 0
        _count = 0
        for result in results["count"]:
            _count += result
        for result in results["sum"]:
            _sum += result
        if _count == 0:
            return None
        else:
            return _sum / _count

    @abstractmethod
    def get_results(self,metadata, agg_funcs: List[str], function, dataset: str,  my_mapping: Dict[str,str]) -> \
    Dict[str, List[Number]]:
        pass
