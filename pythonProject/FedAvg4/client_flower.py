"""pandas_example: A Flower / Pandas app."""

import warnings

from sympy import symbols, sympify

from flwr.client import ClientApp
from flwr.common import Context, Message, MetricsRecord, RecordSet
from flwr.client.typing import ClientFnExt, Mod

from typing import Optional,List,Dict

from server_flower import PARAMS,AGG
from dataset_pandas import PandasDataset

fds = None  # Cache FederatedDataset

warnings.filterwarnings("ignore", category=UserWarning)


def list_to_map(lst: List[str])->Dict[str, str]:
    out = {}
    for key,value in zip(lst[::2], lst[1::2]):
        out[key]=value
    return out

class MyClientApp(ClientApp):



    def __init__(
        self,
        client_fn: Optional[ClientFnExt] = None,  # Only for backward compatibility
        mods: Optional[list[Mod]] = None,
    ) -> None:
        super().__init__(client_fn,mods)
        self.AGG_FUNC={AGG.SUM.__str__():self.local_sum, AGG.COUNT.__str__():self.local_count}


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
                if label==PARAMS.MAPPING.__str__():
                    mapping=list_to_map(value)
                elif label==PARAMS.COL_FUNC.__str__():
                    function_string=value
                elif label == PARAMS.AGG_FUNC.__str__():
                    if value==AGG.AVG.__str__():
                        agg_functions.append(AGG.SUM.__str__())
                        agg_functions.append(AGG.COUNT.__str__())
                    else:
                        agg_functions.append(value)
            out={}
            for agg_func in agg_functions:
                out[agg_func]=self.AGG_FUNC[agg_func](dataset,function_string, mapping)
            reply_content = RecordSet(metrics_records={PARAMS.RESULTS.__str__(): MetricsRecord(out)})
            return msg.create_reply(reply_content)


    def get_clientapp_dataset(self,partition_id: int, num_partitions: int):
        return PandasDataset(num_partitions=10, partition_id=2)
        # # Only initialize `FederatedDataset` once
        # global fds
        # if fds is None:
        #     partitioner = IidPartitioner(num_partitions=num_partitions)
        #     fds = FederatedDataset(
        #         dataset="scikit-learn/iris",
        #         partitioners={"train": partitioner},
        #     )
        # dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
        # print("---<",dataset.columns)
        # return dataset

    def local_sum(self,dataset,function_string,  features):
        return dataset.local_sum(function_string, features)

    def local_count(self,dataset,function_string,  features):
        return dataset.local_count(function_string,  features)


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