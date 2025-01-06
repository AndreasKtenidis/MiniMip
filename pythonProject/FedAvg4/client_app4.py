"""pandas_example: A Flower / Pandas app."""

import warnings

import numpy as np
from datasets.packaged_modules.pandas import pandas
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

from flwr.client import ClientApp
from flwr.common import Context, Message, MetricsRecord, RecordSet

from flwr.client.typing import ClientFnExt, Mod

from typing import Optional

fds = None  # Cache FederatedDataset

warnings.filterwarnings("ignore", category=UserWarning)



class MyClientApp(ClientApp):
    AGG_FUNC = {}

    def __init__(
        self,
        client_fn: Optional[ClientFnExt] = None,  # Only for backward compatibility
        mods: Optional[list[Mod]] = None,
    ) -> None:
        super().__init__()
        self.AGG_FUNC["AGG_SUM"]=self.local_sum

        @self.query()
        def query(msg: Message, context: Context):
            # Read the node_config to fetch data partition associated to this node
            partition_id = context.node_config["partition-id"]
            num_partitions = context.node_config["num-partitions"]

            dataset = self.get_clientapp_dataset(partition_id, num_partitions)

            # print("--------->",msg.content.configs_records["my_config"])
            print(msg.content.configs_records["my_config"])
            for my_func, features in msg.content.configs_records["my_config"].items():
                print(my_func)
                out = {}
                for feature in features:
                    out[feature] = self.AGG_FUNC[my_func](dataset, feature)
                print(out)

            reply_content = RecordSet(metrics_records={"query_results": MetricsRecord(out)})
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
        # Use just the specified columns
        print("->",partition_id,"/",num_partitions)
        return dataset[["SepalLengthCm", "SepalWidthCm"]]

    def local_sum(self,dataset, feature_name):
        return dataset[feature_name].sum()



# Flower ClientApp
app = MyClientApp()


