"""pandas_example: A Flower / Pandas app."""

import warnings
from numbers import Number

from flwr.client import ClientApp
from flwr.common import Context, Message, MetricsRecord, RecordSet
from typing import Optional, List, Dict


from flwr.client.typing import ClientFnExt, Mod

from pythonProject.FedAvg5.interface.ClientInterface import ClientInterface

fds = None  # Cache FederatedDataset

warnings.filterwarnings("ignore", category=UserWarning)



class MyClientApp(ClientApp,ClientInterface):

    def __init__(
        self,
        client_fn: Optional[ClientFnExt] = None,  # Only for backward compatibility
        mods: Optional[list[Mod]] = None,
    ) -> None:
        super().__init__(client_fn,mods)



        @self.query()
        def query(msg: Message, context: Context):
            out={}
            print("!!!!!!!!!!!!")
            print(context)
            # Read the node_config to fetch data partition associated to this node
            # partition_id = context.node_config["partition-id"]
            # num_partitions = context.node_config["num-partitions"]

            # dataset = self.get_clientapp_dataset(partition_id, num_partitions)

            # print("--------->",msg.content.configs_records["my_config"])
            # print(msg.content.configs_records["my_config"])

            # agg_func_list=[]
            # function=None
            # my_mapping={}
            # dataset=None
            # for key, values in msg.content.configs_records["my_config"].items():
            #     if key=="DATASET":
            #         dataset  = values.pop(0)
            #     elif key=="COL_FUNC":
            #         function = values.pop(0)
            #     elif key == "AGG_FUNC":
            #         agg_func_list= values
            #     elif key == "MAPPING":
            #         my_mapping= list_to_map(values)
            # self.get_result(agg_func_list,function,dataset,my_mapping)
            #
            #
            # reply_content = RecordSet(metrics_records={"query_results": MetricsRecord(out)})
            # return msg.create_reply(reply_content)




    def local_sum(self, function, dataset, mapping) -> Number:
        return 0

    def local_count(self, dataset, mapping) -> int:
        return 0

def list_to_map(lst: List[str])->Dict[str, str]:
    out = {}
    for key,value in zip(lst[::2], lst[1::2]):
        out[key]=value
    return out