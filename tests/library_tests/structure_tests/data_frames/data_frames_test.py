import pandas as pd

from tests.help_datasets.dummy_dataset import DummyDataset
from library.templates.partitioned_table import PartitionedPandasTable
from system.client.aggregation_client import PandasAggClient
from tests.test_template.test_template import FederationTestTemplate



class FedDataFramesTest(FederationTestTemplate):
    def federated_computation(self, local_dataset):
        agg = PandasAggClient(self.client)
        return (agg.global_min(local_dataset),
                agg.global_max(local_dataset),
                agg.global_count(local_dataset),
                agg.global_avg(local_dataset))

    def centralized_computation(self, df):
        return (df.min(),
                df.max(),
                df.count(),
                df.mean())

    def get_partitioned_pandas_table(self) -> PartitionedPandasTable:
        return DummyDataset()

    def compare(self, federated_output, global_output):
        print(federated_output)
        print(global_output)