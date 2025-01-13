from abc import ABC, abstractmethod
from typing import Dict, List,Optional
from numbers import Number


class ClientInteface(ABC):

    def agg_sum(self, node_ids: List[int], function, dataset: str,  mapping: Dict[str:str]) -> Number:
        results = self.get_results(["sum"], node_ids, function, dataset, mapping)
        _sum = 0
        for result in results["sum"]:
            _sum += result
        return _sum

    def agg_count(self, node_ids: List[int], dataset: str, mapping: Dict[str:str]) -> int:
        results = self.get_results(["count"], node_ids, None, dataset, mapping)
        _count = 0
        for result in results["count"]:
            _count += result
        return _count

    def agg_avg(self, node_ids: List[int], function, dataset: str, mapping: Dict[str:str]) -> Optional[Number]:
        results = self.get_results(["sum", "count"], node_ids, function, dataset, mapping)
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
    def get_results(self, agg_funcs: List[str], node_ids: List[int], function, dataset: str,  mapping: Dict[str:str]) -> \
    Dict[str, List[Number]]:
        pass
