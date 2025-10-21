from tests_and_experiments.datasets.job_training import JobTrainingDataset
from tests_and_experiments.library_tests.propensity_score.propensity_score_test import PropensityScoreTest



PropensityScoreTest(0, 2,
                        dataset=JobTrainingDataset(),
                        operation_id=9,
                        confounders = ['age', 'educ', 'married', 'nodegree', 're74', 're75', 'black', 'hispan', 'white'],
                        treatment = 'treat')

