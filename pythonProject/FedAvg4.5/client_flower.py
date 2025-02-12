import warnings

from collections import OrderedDict


from flwr.client import ClientApp
from flwr.common import Context, Message, MetricsRecord, RecordSet, ParametersRecord, array_from_numpy

from flwr.client.typing import ClientFnExt, Mod
from typing import Optional, Callable,List
import inspect
import json

from alg1 import algorithm

from numpy import ndarray

from _agg_function import AggFunction

from _constants import PARAMS
from dataset_pandas import PandasDataset


from _client_server import LocalStorage


fds = None  # Cache FederatedDataset

warnings.filterwarnings("ignore", category=UserWarning)

class MyClientApp(ClientApp):
    def __init__(
        self,
        client_fn: Optional[ClientFnExt] = None,  # Only for backward compatibility
        mods: Optional[list[Mod]] = None,
    ) -> None:
        super().__init__(client_fn,mods)

        @self.query()
        def query(msg: Message, context_: Context):
            # Fetching variables from the message and its context
            partition_id,num_partitions,node_id = MyClientApp.get__context(context_)
            fed_round =MyClientApp.get_round(msg)
            if fed_round==0:
                mapping_string, dataset_name, function_string = MyClientApp.get_configs_first(msg)
                return self.first_round(context_,msg ,fed_round , mapping_string, partition_id, num_partitions)
            else:
                mapping_string = MyClientApp.get_mappings(msg)
                return self.next_round(context_,msg ,fed_round , mapping_string)
            # Getting the Dataset and mapping attributes to variables



    @staticmethod
    def first_round(context_: Context, msg:Message, fed_round:int, mapping_string:str, partition_id:int, num_partitions:int):
        mapping = json.loads(mapping_string)
        dataset = MyClientApp.get_clientapp_dataset(partition_id, num_partitions)
        local_input = {}
        for key, value in mapping.items():
            local_input[key] = dataset.get_attribute(value)

        # Getting and executing the function
        alg = algorithm
        func: Callable[[LocalStorage, ndarray], List[AggFunction]] = alg.get_operation(fed_round)
        aggregations: List[AggFunction] = MyClientApp.map_and_execute(func, FlowerStorage(context_), local_input)

        # Sending result of aggregation round
        return MyClientApp.reply(aggregations, msg)

    @staticmethod
    def next_round(context_: Context, msg:Message, fed_round:int, mapping_string:str):
        var_mapping = json.loads(mapping_string)
        # Getting and executing the function
        alg = algorithm

        func: Callable[[LocalStorage, ndarray], List[AggFunction]] = alg.get_operation(fed_round)
        aggregations: List[AggFunction] = MyClientApp.map_and_execute(func, FlowerStorage(context_), var_mapping)

        # Sending result of aggregation round
        return MyClientApp.reply(aggregations, msg)


    @staticmethod
    def reply(aggregations:List[AggFunction], msg: Message):
        _metrics_records = {}
        for agg_func in aggregations:
            out = {agg_func.get_operation().value:agg_func.get_local_aggregations()}
            _metrics_records[agg_func.get_var_name()] = MetricsRecord(out)
        reply_content = RecordSet(metrics_records=_metrics_records)
        return msg.create_reply(reply_content)

    @staticmethod
    def get__context(context_: Context):
        return context_.node_config["partition-id"],context_.node_config["num-partitions"],context_.node_id

    @staticmethod
    def get_configs_first(msg: Message):
        configs = msg.content.configs_records[PARAMS.OPERATION_ID.value]
        return (configs[PARAMS.MAPPING.value],
                configs[PARAMS.DATASET.value],
                configs[PARAMS.FUNCTION.value])

    @staticmethod
    def get_mappings(msg: Message):
        configs = msg.content.configs_records[PARAMS.OPERATION_ID.value]
        return (
                configs[PARAMS.MAPPING.value])

    @staticmethod
    def get_round(msg: Message):
        configs = msg.content.configs_records[PARAMS.OPERATION_ID.value]
        return configs[PARAMS.ROUND.value]

    @staticmethod
    def get_clientapp_dataset(partition_id: int, num_partitions: int):
        return PandasDataset(partition_id=partition_id, num_partitions=num_partitions)

    @staticmethod
    def map_and_execute(func:Callable[[LocalStorage, ndarray], List[AggFunction]],client:LocalStorage, data_dict)->List[AggFunction]:
        # Get function parameters
        params = inspect.signature(func).parameters
        param_names = params.keys()

        # Extract relevant arguments from the dictionary
        mapped_args = {param: data_dict[param] for param in param_names if param in data_dict}

        # Execute the function with the mapped arguments
        return func(client,**mapped_args)



class FlowerStorage(LocalStorage):

    def __init__(self, context_: Context):
        self.context = context_

    def load(self, x: str):
        output = self.context.state.parameters_records["my_parameters"].get(x)
        return output.numpy()


    def store(self, key: str, value):
        arr = array_from_numpy(value)
        if "my_parameters" not in self.context.state.parameters_records:
            self.context.state.parameters_records["my_parameters"]= ParametersRecord(OrderedDict({key: arr}))




# Flower ClientApp
app = MyClientApp()