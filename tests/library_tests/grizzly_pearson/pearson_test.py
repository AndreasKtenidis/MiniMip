from library.stats.bivariate_statistics import PearsonCorrelationGrizzly
from mini_mip_system.client.grpc_agg_client import GRPCClient
from tests.test_template.grizzly_test_template import GrizzlyFederationTestTemplate

class PearsonTest(GrizzlyFederationTestTemplate):
    def federated_computation(self,conn, local_dataset):
        pearson_value = PearsonCorrelationGrizzly(self.client).compute(local_dataset, x='x', y='y')
        return pearson_value

    def centralized_computation(self, centralized_dataset):
        return None

    def compare(self, federated_output, global_output):
        print("federated_output:", federated_output)

