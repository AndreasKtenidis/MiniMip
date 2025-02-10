"""pandas_example: A Flower / Pandas app."""
import json
import warnings

from flwr.client import ClientApp
from flwr.common import Context, Message, MetricsRecord, RecordSet
from flwr.client.typing import ClientFnExt, Mod
import time
from typing import Optional


from flask_communicator import FlaskCommunicator
from constants import PARAMS,AGG
from dataset_pandas import PandasDataset
import inspect

from federator import NumpyAggregatorClient
from abstract_function import MeanSquare

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
        def query(msg: Message, context: Context):
            # Fetching variables from the message and its context
            partition_id,num_partitions,node_id = MyClientApp.get_context(context)
            operation_id,mapping_string,dataset_name,function_string = MyClientApp.get_configs(msg)

            # Creating the Aggregator
            aggregator: FlowerNumpyAggregatorClient = FlowerNumpyAggregatorClient(node_id, num_partitions,
                                                                                  operation_id)
            # Getting the Dataset and mapping attributes to variables
            mapping = json.loads(mapping_string)
            dataset = MyClientApp.get_clientapp_dataset(partition_id,num_partitions)
            local_input={}
            for key,value in mapping.items():
                local_input[key] = dataset.get_attribute(value)

            # Getting and executing the function
            func = MeanSquare(aggregator)
            answer = MyClientApp.map_and_execute(func.compute, local_input)

            # Printing the answer
            out = {'answer':float(answer)}
            reply_content = RecordSet(metrics_records={PARAMS.RESULTS.__str__(): MetricsRecord(out)})
            return msg.create_reply(reply_content)

    @staticmethod
    def get_context(context: Context):
        return context.node_config["partition-id"],context.node_config["num-partitions"],context.node_id

    @staticmethod
    def get_configs(msg: Message):
        configs = msg.content.configs_records[PARAMS.OPERATION_ID.value]
        return (configs[PARAMS.OPERATION_ID.value],
                configs[PARAMS.MAPPING.value],
                configs[PARAMS.DATASET.value],
                configs[PARAMS.FUNCTION.value])

    @staticmethod
    def get_clientapp_dataset(partition_id: int, num_partitions: int):
        return PandasDataset(partition_id=partition_id, num_partitions=num_partitions)

    @staticmethod
    def map_and_execute(func, data_dict):
        # Get function parameters
        params = inspect.signature(func).parameters
        param_names = params.keys()
        # Extract relevant arguments from the dictionary
        mapped_args = {param: data_dict[param] for param in param_names if param in data_dict}
        # Execute the function with the mapped arguments
        return func(**mapped_args)


class FlowerNumpyAggregatorClient(NumpyAggregatorClient):

    def __init__(self, node_id:int, client_count, operation_id:int):
        super().__init__()
        self.communicator = FlaskCommunicator()
        self.node_id = node_id
        self.operation_id = operation_id
        self.agg_round=0
        self.client_count=client_count

    def __global_sum__(self, local_sum2) -> float:
        self.communicator.add_aggregation(self.operation_id, self.node_id, self.agg_round, AGG.SUM, local_sum2)
        while 1 == 1:
            answer = self.communicator.get_aggregation(self.operation_id, self.agg_round, AGG.SUM.value,
                                                       self.client_count)
            if answer == 'null' or (answer is None) or answer == '':
                time.sleep(1)
            else:
                self.agg_round+=1
                return answer

    def __global_count__(self, local_count2) -> int:
        self.communicator.add_aggregation(self.operation_id, self.node_id, self.agg_round, AGG.COUNT.value, local_count2)
        while 1==1:
            answer = self.communicator.get_aggregation(self.operation_id,self.agg_round,AGG.COUNT.value,self.client_count)
            if answer=='null' or (answer is None) or answer=='':
                time.sleep(1)
            else:
                self.agg_round += 1
                return answer

    def __global_avg__(self, local_sum, local_count) -> int:
        self.communicator.add_aggregation(self.operation_id,self.node_id,self.agg_round,AGG.SUM,local_sum)
        self.communicator.add_aggregation(self.operation_id, self.node_id, self.agg_round, AGG.COUNT.value,
                                          local_count)
        while 1==1:
            answer = self.communicator.get_aggregation(self.operation_id,self.agg_round,AGG.AVG.value,self.client_count)
            if answer=='null' or (answer is None) or answer=='':
                time.sleep(1)
            else:
                self.agg_round += 1
                return answer

# Flower ClientApp
app = MyClientApp()