from library.causal.ipwt import IPWT
import numpy as np
import pandas as pd

from tests.help_datasets.titanic import Titanic
from tests.help_datasets.titanic_as_disease import TitanicAsDisease
from library.templates.partitioned_table import PartitionedPandasTable
from tests.test_template.test_template import FederationTestTemplate
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact

from library.group_comparisons.chi_squared import ChiSquared
from library.group_comparisons.fisher_exact import FisherExact

class ChiSquaredAndFisherTest(FederationTestTemplate):

    def centralized_computation(self, centralized_dataset):
        def chi_test(df):
            # Inspect
            print(df.head())

            # Features and target
            # Create a contingency table
            contingency_table = pd.crosstab(df['Pclass'], df['Survived'])
            # Run Chi-Squared test
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            return chi2, p, dof, expected

        def fisher_test(df):
            table = pd.crosstab(df['Sex'], df['Survived'])
            oddsratio, p_value = fisher_exact(table)
            return oddsratio, p_value

        return chi_test(centralized_dataset), fisher_test(centralized_dataset)

    def federated_computation(self, local_dataset):
        chi2, p, dof, expected = ChiSquared(self.client).compute(local_dataset, factor='Pclass', outcome='Survived')

        print("\nChi-squared Statistic:", chi2)
        print("p-value:", p)
        print("Degrees of Freedom:", dof)
        print("-------------------------")
        oddsratio, p_value = FisherExact(self.client).compute(local_dataset, factor='Sex', outcome='Survived')
        return chi2, p, dof, expected,oddsratio, p_value

    def get_partitioned_pandas_table(self) -> PartitionedPandasTable:
        return Titanic()

    def compare(self, federated_output, global_output):
        print(federated_output)
        print(global_output)




