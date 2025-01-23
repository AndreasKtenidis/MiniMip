"""pandas_example: A Flower / Pandas app."""

import warnings
from abc import ABC

from sympy import symbols, sympify

from flwr.client import ClientApp
from flwr.common import Context, Message, MetricsRecord, RecordSet
from flwr.client.typing import ClientFnExt, Mod
import time
from typing import Optional,List,Dict
import random
from flask_communicator import Flask_Communicator
from server_flower import PARAMS,AGG
from dataset_pandas import PandasDataset

from flwr.common.logger import log
from logging import INFO

import numpy as np

from federator import NumpyClient

fds = None  # Cache FederatedDataset

warnings.filterwarnings("ignore", category=UserWarning)


def list_to_map(lst: List[str])->Dict[str, str]:
    out = {}
    for key,value in zip(lst[::2], lst[1::2]):
        out[key]=value
    return out


def local_count(dataset, function_string, features):
    return dataset.local_count(function_string,  features)


def local_sum(dataset, function_string, features):
    return dataset.local_sum(function_string, features)


def get_clientapp_dataset(partition_id: int, num_partitions: int):
    return PandasDataset(partition_id=partition_id,num_partitions=num_partitions)

class MyClientApp(ClientApp):

    def __init__(
        self,
        client_fn: Optional[ClientFnExt] = None,  # Only for backward compatibility
        mods: Optional[list[Mod]] = None,
    ) -> None:
        super().__init__(client_fn,mods)

        @self.query()
        def query(msg: Message, context: Context):

            node_id = context.node_id
            log(INFO, "Calling on ")
            print(log)
            partition_id = context.node_config["partition-id"]
            num_partitions = context.node_config["num-partitions"]
            #
            operation_id = 666
            numpy_client:FlowerNumpyClient = FlowerNumpyClient(node_id,num_partitions,operation_id)
            x = np.random.random(10)
            y = np.random.random(10)
            # Read the node_config to fetch data partition associated to this node

            dataset = get_clientapp_dataset(partition_id, num_partitions)
            print("------>",numpy_client.count(x**2))

            out = {}
            reply_content = RecordSet(metrics_records={PARAMS.RESULTS.__str__(): MetricsRecord(out)})
            return msg.create_reply(reply_content)

class FlowerNumpyClient(NumpyClient):
    def __init__(self, node_id:int, client_count, operation_id:int):
        super().__init__()
        self.communicator = Flask_Communicator()
        self.node_id = node_id
        self.operation_id = operation_id
        self.agg_round=0
        self.client_count=client_count

    def __global_sum__(self, local_sum2) -> float:
        self.communicator.add_aggregation(self.operation_id,
                                          self.node_id,
                                          self.agg_round,
                                          "sum",
                                          local_sum2)
        while 1 == 1:
            answer = self.communicator.get_aggregation(self.operation_id, self.agg_round, "count", self.client_count)
            if answer == 'null' or (answer is None) or answer == '':
                time.sleep(1)
            else:
                return answer

    def __global_count__(self, local_count2) -> int:
        self.communicator.add_aggregation(self.operation_id,
                                          self.node_id,
                                          self.agg_round,
                                          "count",
                                          local_count2)
        while 1==1:
            answer = self.communicator.get_aggregation(self.operation_id,self.agg_round,"count",self.client_count)
            if answer=='null' or (answer is None) or answer=='':
                time.sleep(1)
            else:
                return answer

def replace_variables(expression, mapping):
    """
    Replaces variables in a mathematical expression with new variables based on a mapping.
    """
    expr = sympify(expression)
    symbol_mapping = {symbols(k): symbols(v) for k, v in mapping.items()}
    updated_expr = expr.subs(symbol_mapping)
    return str(updated_expr)


# Flower ClientApp
app = MyClientApp()