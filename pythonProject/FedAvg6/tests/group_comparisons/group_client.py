
from library.stat_models.ordinal_logistic_regression import FedOrdinalLogisticRegression
from system.client.grpc_agg_client import GRPCClient
import pandas as pd
from data.experiment_datasets.pandas_datasets.federated_dataset import FederatedPandasDataset
from sklearn.metrics import classification_report
from library.group_comparisons.chi_squared import ChiSquared
from library.group_comparisons.fisher_exact import FisherExact

class TmpDataset(FederatedPandasDataset):

    def get_dataset(self) -> pd.DataFrame:
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
        return df


config = {
        'num_clients':2
    }

def compute(client_num):
    # # Creating Client
    client = GRPCClient(client_num, config['num_clients'], client_num)
    df = TmpDataset(client_num,config['num_clients']).get_local_dataset()

    chi2, p, dof, expected = ChiSquared(client).compute(df,factor= 'Pclass',outcome='Survived')

    print("\nChi-squared Statistic:", chi2)
    print("p-value:", p)
    print("Degrees of Freedom:", dof)
    print("-------------------------")
    oddsratio, p_value = FisherExact(client).compute(df,factor= 'Sex',outcome='Survived')

    print("\nOdds Ratio:", oddsratio)
    print("p-value:", p_value)


