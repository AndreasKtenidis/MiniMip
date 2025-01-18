from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset
from sympy import symbols, sympify

from dataset_abstract import Dataset


class PandasDataset(Dataset):
    # Static attribute to store the global fds object
    fds = None

    def __init__(self,partition_id, num_partitions):
        # Initialize the dataset only once
        if PandasDataset.fds is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            PandasDataset.fds = FederatedDataset(
                dataset="scikit-learn/iris",
                partitioners={"train": partitioner}
            )

        # Load a specific partition and format it as pandas DataFrame
        self.dataset = PandasDataset.fds.load_partition(partition_id, "train").with_format("pandas")[:]

    def get_data(self):
        """Returns the dataset loaded into pandas format."""
        return self.dataset

    def local_sum(self, function_string, mapping):
        d_mapping={}
        for k in mapping.keys():
            d_mapping[k]="self.dataset['"+mapping[k]+"']"
        expression = replace_variables(function_string, d_mapping)
        expression = "(" + expression + ").sum()"
        return eval(expression)

    def local_count(self, function_string, mapping):
        return self.dataset[get_variable(mapping)].count()+0.0

def replace_variables(expression, mapping):
    """
    Replaces variables in a mathematical expression with new variables based on a mapping.
    """
    expr = sympify(expression)
    symbol_mapping = {symbols(k): symbols(v) for k, v in mapping.items()}
    updated_expr = expr.subs(symbol_mapping)
    return str(updated_expr)

def get_variable(mapping):
    for value in mapping.values():
        return value
