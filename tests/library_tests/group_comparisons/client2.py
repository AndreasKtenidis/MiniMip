from tests.help_datasets.titanic import TitanicDataset
from tests.library_tests.group_comparisons.chisquared_fisher_test import ChiSquaredAndFisherTest

def main():
    ChiSquaredAndFisherTest(1, 2, dataset =TitanicDataset(),operation_id=123)

if __name__ == "__main__":
    main()