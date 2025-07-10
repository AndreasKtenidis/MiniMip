import numpy as np

class CoefficientTest:
    # Compute the sum of absolute differences (L1 distance) between two coefficient vectors
    @staticmethod
    def coeff_absolute_difference(*, federated_coef, centralized_coef) -> float:
        return np.sum(np.abs(centralized_coef - federated_coef))

    # Compute the Euclidean distance (L2 norm) between two coefficient vectors
    @staticmethod
    def coeff_euclidean_distance(*, federated_coef, centralized_coef) -> float:
        return np.linalg.norm(centralized_coef - federated_coef)

    # Compute the sum of relative differences between two coefficient vectors, with epsilon for stability
    @staticmethod
    def coeff_relative_distance(*, federated_coef, centralized_coef, epsilon=1e-8) -> float:
        return np.sum(np.abs(centralized_coef - federated_coef) / (np.abs(centralized_coef) + epsilon))

    # Placeholder method if you want to add batch comparison logic later
    def compute(self, *args, **kwargs):
        pass
