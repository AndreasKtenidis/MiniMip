from library.causal.ipwt import IPWT
from system.client.grpc_agg_client import GRPCClient
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

from tests.propensity_score.test import propensity_score_non_fd
from tests.testing_dataset.partitioned_table import PartitionedPandasTable
import statsmodels.api as sm
from library.causal.propensity_score2 import PropensityScore

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
    data =dataset.get_local_dataset(client_num,config['num_clients'])

    # Propensity score matching:
    propensity_score = PropensityScore(client)
    matched_control, matched_df = propensity_score.compute(data, features=TmpDataset.features, treatment='treat')
    print(matched_control)
    print(matched_df)
    # Creating a global output from local outputs
    nfd_matched_control,nfd_matched_df = propensity_score_non_fd(TmpDataset.dataset,features =TmpDataset.features,treatment = 'treat')



dataset = TmpDataset()
config = {'num_clients':2}


