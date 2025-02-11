"""pandas_example: A Flower / Pandas app."""
import json
import warnings

from flwr.client import ClientApp
from flwr.common import Context, Message, MetricsRecord, RecordSet
from flwr.client.typing import ClientFnExt, Mod
import time
from typing import Optional

from constants import PARAMS,AGG
from dataset_pandas import PandasDataset
import inspect

from client_server import NumpyAggregatorClient


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
            partition_id,num_partitions,node_id = MyClientApp.get_context(context)
            operation_id,mapping_string,dataset_name,function_string = MyClientApp.get_configs(msg)

            # Creating the Aggregator

            # Getting the Dataset and mapping attributes to variables
            mapping = json.loads(mapping_string)
            dataset = MyClientApp.get_clientapp_dataset(partition_id,num_partitions)
            local_input={}
            for key,value in mapping.items():
                local_input[key] = dataset.get_attribute(value)

            # Getting and executing the function
            answer={}
            # Printing the answer
            out = {'answer':float(answer)}
            reply_content = RecordSet(metrics_records={PARAMS.RESULTS.__str__(): MetricsRecord(out)})
            return msg.create_reply(reply_content)

    def store(self, key: str, value):
        pass

    def load(self, x: str):
        pass

    @staticmethod
    def get_context(context: Context):
        print(context)
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





# Flower ClientApp
app = MyClientApp()