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
        self.AGG_FUNC["AGG_COUNT"] = self.local_count

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
        return dataset[["SepalLengthCm", "SepalWidthCm"]]

    def local_sum(self,function_string,dataset, features):
        mapping={'x':"dataset['SepalLengthCm']",'y':"dataset['SepalWidthCm']"}
        expression = replace_variables_in_order(function_string, mapping)
        expression="("+expression+").sum()"
        return eval(expression)

    def local_count(self,function_string,dataset, features):
        return dataset['SepalLengthCm'].count()+0.0

# Function to replace variables in an AST
class OrderedVariableReplacer(ast.NodeTransformer):
    def __init__(self, replacements):
        self.replacements = replacements
        self.seen = set()  # Keep track of already replaced variables

    def visit_Name(self, node):
        # If the variable is in the replacements and not yet replaced
        if node.id in self.replacements and node.id not in self.seen:
            self.seen.add(node.id)  # Mark this variable as replaced
            # Replace with the corresponding value (parsed into AST)
            return ast.parse(str(self.replacements[node.id]), mode='eval').body
        return node  # Return unchanged if not in replacements or already replaced


def replace_variables_in_order(expression, replacements):
    # Parse the expression into an AST
    parsed_expr = ast.parse(expression, mode='eval')

    # Replace variables using the OrderedVariableReplacer
    replacer = OrderedVariableReplacer(replacements)
    new_expr_ast = replacer.visit(parsed_expr)

    # Compile the modified AST back into a string expression
    return ast.unparse(new_expr_ast)


# Flower ClientApp
app = MyClientApp()


