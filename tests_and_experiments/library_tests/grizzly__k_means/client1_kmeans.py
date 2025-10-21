from tests_and_experiments.library_tests.grizzly__k_means.k_means import KMeansGrizzlyTest
from tests_and_experiments.datasets.blob import BlobDataset


KMeansGrizzlyTest(0, 2,
                           dataset=BlobDataset(),
                           operation_id=0)