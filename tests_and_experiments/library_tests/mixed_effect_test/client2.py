from tests_and_experiments.datasets.job_training import JobTrainingDataset
from tests_and_experiments.datasets.titanic_as_disease import TitanicAsDiseaseDataset
from tests_and_experiments.library_tests.mixed_effect_test.test_federated_mixed_effects import Test_FederatedMixedEffect
from tests_and_experiments.library_tests.propensity_score.propensity_score_test import PropensityScoreTest



Test_FederatedMixedEffect(1, 2,
                          patients=TitanicAsDiseaseDataset(),
                          # covariates = ['age', 'married', 'nodegree', 're74', 're75', 'black', 'hispan', 'white'],
                          covariates=['age', 'sibsp', 'parch', 'fare'],
                          center='pclass',
                          outcome='survived')

