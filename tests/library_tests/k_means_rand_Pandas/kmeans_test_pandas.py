from tests.test_template.test_template import FederationTestTemplate


class KmeansPandasTest(FederationTestTemplate):

    def __init__(self, client_id, client_count, *, dataset, operation_id=0,aggregation_server="localhost:50051", features):
        self.features = features
        super().__init__(client_id, client_count, dataset=dataset, operation_id=operation_id,aggregation_server=aggregation_server)

    def federated_computation(self, local_dataset):
        from library.working_in_minimip.pandas import KMeansPandas
        computer = KMeansPandas(self.client)
        output = computer.compute(local_dataset,3)
        return output

    def centralized_computation(self, centralized_dataset):
        return None

    def compare(self, federated_output, global_output):
        print(federated_output)
        print("No global output",global_output)

