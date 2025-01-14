"""pandas_example: A Flower / Pandas app."""

import warnings


import ast

import numpy as np
from datasets.packaged_modules.pandas import pandas
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

from flwr.client import ClientApp
from flwr.common import Context, Message, MetricsRecord, RecordSet

from flwr.client.typing import ClientFnExt, Mod

from typing import Optional,List,Dict

from pyarrow.dataset import dataset

from server_app4 import AGG
from server_app4 import PARAMS

fds = None  # Cache FederatedDataset

warnings.filterwarnings("ignore", category=UserWarning)


def list_to_map(lst: List[str])->Dict[str, str]:
    out = {}
    for key,value in zip(lst[::2], lst[1::2]):
        out[key]=value
    return out

class MyClientApp(ClientApp):

    AGG_FUNC = {}

    def __init__(
        self,
        client_fn: Optional[ClientFnExt] = None,  # Only for backward compatibility
        mods: Optional[list[Mod]] = None,
    ) -> None:
        super().__init__(client_fn,mods)
        self.AGG_FUNC[AGG.SUM] = self.local_sum
        self.AGG_FUNC[AGG.COUNT] = self.local_count

        @self.query()
        def query(msg: Message, context: Context):
            # Read the node_config to fetch data partition associated to this node
            partition_id = context.node_config["partition-id"]
            num_partitions = context.node_config["num-partitions"]

            dataset = self.get_clientapp_dataset(partition_id, num_partitions)
            print(msg.content.configs_records["my_config"])
            mapping={}
            function_string=""
            agg_functions=[]
            for label, value in msg.content.configs_records["my_config"].items():
                if label==PARAMS.MAPPING:
                    mapping=list_to_map(value)
                elif label==PARAMS.COL_FUNC:
                    function_string=value
                elif label == PARAMS.AGG_FUNC:
                    if value==AGG.AVG:
                        agg_functions.append(AGG.SUM)
                        agg_functions.append(AGG.COUNT)
                    else:
                        agg_functions.append(value)
            out={}
            for agg_func in agg_functions:
                out[agg_func]=self.AGG_FUNC[agg_func](function_string,dataset, mapping)
            reply_content = RecordSet(metrics_records={PARAMS.RESULTS: MetricsRecord(out)})
            return msg.create_reply(reply_content)


    def get_clientapp_dataset(self,partition_id: int, num_partitions: int):
        # Only initialize `FederatedDataset` once
        global fds
        if fds is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            fds = FederatedDataset(
                dataset="scikit-learn/iris",
                partitioners={"train": partitioner},
            )
        dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
        return dataset[["SepalLengthCm", "SepalWidthCm"]]

    def local_sum(self,function_string, dataset, features):
        print("--->",features)
        mapping={'x':"dataset['SepalLengthCm']",'y':"dataset['SepalWidthCm']"}
        expression = replace_variables(function_string, mapping)
        expression="("+expression+").sum()"
        return eval(expression)

    def local_count(self,function_string, dataset, features):
        print("--->",features)
        return dataset['SepalLengthCm'].count()+0.0



from sympy import symbols, sympify


def replace_variables(expression, mapping):
    """
    Replaces variables in a mathematical expression with new variables based on a mapping.
    """
    expr = sympify(expression)
    # Convert mapping keys and values to sympy symbols
    symbol_mapping = {symbols(k): symbols(v) for k, v in mapping.items()}
    # Perform the replacement
    updated_expr = expr.subs(symbol_mapping)
    # Convert back to string for the result
    return str(updated_expr)


# Flower ClientApp
app = MyClientApp()


