from system.client.grpc_agg_client import GRPCClient
import pandas as pd


from library.templates.partitioned_table import PartitionedPandasTable
import statsmodels.api as sm

from library.stats.median import MedianBasedOnHistogram


class TmpDataset(PartitionedPandasTable):

    dataset =sm.datasets.get_rdataset("lalonde", "MatchIt").data
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    features = ['age', 'educ', 'married', 'nodegree', 're74', 're75'] + \
               [col for col in dataset.columns if col.startswith('race_')]

    def get_dataset(self) -> pd.DataFrame:
        return TmpDataset.dataset

def compute(client_num):

    # Creating Client
    client:GRPCClient = GRPCClient(client_num, config['num_clients'], client_num)
    # Federated Dataset
    data = TmpDataset().get_local_dataset(client_num,config['num_clients'])
    median_calc = MedianBasedOnHistogram(client)
    print(type(data['age']))
    median = median_calc.compute(data['age'], num_bins=10)
    print(median)
    print(data['age'].median())



dataset = TmpDataset()
config = {'num_clients':2}


