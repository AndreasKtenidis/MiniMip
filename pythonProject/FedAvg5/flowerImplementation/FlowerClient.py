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

from typing import Optional

from pyarrow.dataset import dataset

from pythonProject.FedAvg5.interface.ClientInterface import ClientInterface

fds = None  # Cache FederatedDataset

warnings.filterwarnings("ignore", category=UserWarning)



class FlowerClient(ClientApp,ClientInterface):


    def __init__(
        self,
        client_fn: Optional[ClientFnExt] = None,  # Only for backward compatibility
        mods: Optional[list[Mod]] = None,
    ) -> None:
        super().__init__()

        @self.query()
        def query(msg: Message, context: Context):
            # Read the node_config to fetch data partition associated to this node
            partition_id = context.node_config["partition-id"]
            num_partitions = context.node_config["num-partitions"]

            dataset = self.get_clientapp_dataset(partition_id, num_partitions)

            # print("--------->",msg.content.configs_records["my_config"])
            print(msg.content.configs_records["my_config"])
            for agg_function, features in msg.content.configs_records["my_config"].items():
                print(agg_function)
                function_string=features.pop(0)
                out={"result":self.AGG_FUNC[agg_function](function_string,dataset, features)}
            reply_content = RecordSet(metrics_records={"query_results": MetricsRecord(out)})
            return msg.create_reply(reply_content)


