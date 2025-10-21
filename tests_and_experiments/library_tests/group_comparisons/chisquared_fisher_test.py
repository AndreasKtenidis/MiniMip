import pandas as pd

from library.group_comparisons.chi_squared import ChiSquared
from library.group_comparisons.fisher_exact import FisherExact
from library.utils.numpy_aggregator import NumpyAggregator

from tests_and_experiments.core.test_template import FederationTestTemplate
from scipy.stats import chi2_contingency



class ChiSquaredAndFisherTest(FederationTestTemplate):

    def centralized_computation(self, centralized_dataset):
        def chi_test(df):

            # Features and target
            # Create a contingency table
            contingency_table = pd.crosstab(df['Pclass'], df['Survived'])
            # Run Chi-Squared test
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            return chi2, p, dof, expected


        chi2, p, dof, expected = chi_test(centralized_dataset)
        return chi2, p, dof, expected

    def federated_computation(self, local_dataset):


        aggr = NumpyAggregator(self.client)

        sex_categories = aggr.fed_union(local_dataset['Sex'].values)
        class_categories = aggr.fed_union(local_dataset['Pclass'].values)
        outcome_categories = aggr.fed_union(local_dataset['Survived'].values)
        chi2, p, dof, expected = ChiSquared(self.client).compute(local_dataset,
                                                                 factor='Pclass',
                                                                 factor_categories=class_categories,
                                                                 outcome='Survived',
                                                                 outcome_categories=outcome_categories)
        oddsratio, p_value = FisherExact(self.client).compute(local_dataset, factor='Sex',
                                                              factor_categories=sex_categories,
                                                              outcome='Survived',
                                                              outcome_categories=outcome_categories)
        return chi2, p, dof, expected,oddsratio, p_value


    def compare(self, federated_output, global_output):
        print(federated_output)
        print(global_output)
