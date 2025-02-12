import warnings

from flwr.client import ClientApp
from flwr.common import Context, Message, MetricsRecord, RecordSet
from flwr.client.typing import ClientFnExt, Mod
from typing import Optional, Callable,List
import inspect
import json

from alg1 import algorithmic_steps

from numpy import ndarray

from _agg_function import AggFunction

from _constants import PARAMS
from dataset_pandas import PandasDataset
from _abstract_algorithm import FederatedAlgorithm

from _client_server import NumpyAggregatorClient


fds = None  # Cache FederatedDataset

warnings.filterwarnings("ignore", category=UserWarning)

class MyClientApp(ClientApp,NumpyAggregatorClient):
    def __init__(
        self,
        client_fn: Optional[ClientFnExt] = None,  # Only for backward compatibility
        mods: Optional[list[Mod]] = None,
    ) -> None:
        super().__init__(client_fn,mods)

        @self.query()
        def query(msg: Message, context: Context):
            # Fetching variables from the message and its context
            partition_id,num_partitions,node_id = MyClientApp.get__context(context)
            fed_round,mapping_string,dataset_name,function_string = MyClientApp.get_configs(msg)

            if fed_round==1:
                print('Round',0)
                return self.first_round(msg ,fed_round , mapping_string, partition_id, num_partitions)
            else:
                pass
            # Getting the Dataset and mapping attributes to variables



    def first_round(self,msg:Message ,fed_round:int , mapping_string:str, partition_id:int, num_partitions:int):
        mapping = json.loads(mapping_string)
        dataset = MyClientApp.get_clientapp_dataset(partition_id, num_partitions)
        local_input = {}
        for key, value in mapping.items():
            local_input[key] = dataset.get_attribute(value)

        # Getting and executing the function
        alg = FederatedAlgorithm(algorithmic_steps)
        func: Callable[[NumpyAggregatorClient, ndarray], List[AggFunction]] = alg.get_operation(fed_round)[fed_round]
        aggregations: List[AggFunction] = MyClientApp.map_and_execute(func, self, local_input)

        # Sending result of aggregation round
        return MyClientApp.first_reply(aggregations, msg)


    def store(self, key: str, value):
        print('Testing')
        pass

    def load(self, x: str):
        print('Testing')
        pass

    @staticmethod
    def first_reply(aggregations:List[AggFunction], msg: Message):
        _metrics_records = {}
        for agg_func in aggregations:
            out = {agg_func.get_operation().value:agg_func.get_local_aggregations()}
            _metrics_records[agg_func.get_var_name()] = MetricsRecord(out)
        reply_content = RecordSet(metrics_records=_metrics_records)
        return msg.create_reply(reply_content)

    @staticmethod
    def get__context(context: Context):
        print(context)
        return context.node_config["partition-id"],context.node_config["num-partitions"],context.node_id

    @staticmethod
    def get_configs(msg: Message):
        configs = msg.content.configs_records[PARAMS.OPERATION_ID.value]
        return (configs[PARAMS.ROUND.value],
                configs[PARAMS.MAPPING.value],
                configs[PARAMS.DATASET.value],
                configs[PARAMS.FUNCTION.value])

    @staticmethod
    def get_clientapp_dataset(partition_id: int, num_partitions: int):
        return PandasDataset(partition_id=partition_id, num_partitions=num_partitions)

    @staticmethod
    def map_and_execute(func:Callable[[NumpyAggregatorClient, ndarray], List[AggFunction]],client:NumpyAggregatorClient, data_dict)->List[AggFunction]:
        # Get function parameters
        params = inspect.signature(func).parameters
        param_names = params.keys()

        # Extract relevant arguments from the dictionary
        mapped_args = {param: data_dict[param] for param in param_names if param in data_dict}

        # Execute the function with the mapped arguments
        return func(client,**mapped_args)


# Flower ClientApp
app = MyClientApp()