from library.core.statistical_function import StatisticalFunction
import numpy as np
from library.under_development.stats.univariate_statistics import Variance
from library.utils.numpy_aggregator import NumpyAggregator


class FederatedMixedEffect(StatisticalFunction):
    def compute(self, patients,*,covariates,center,outcome):
        variance  = Variance(self.client)
        # Group by center and iterate through each center
        centers_grouped = patients.groupby(center)
        print(centers_grouped)
        sigma2 = variance.compute(patients[outcome].values)
        for center_id, center_data in centers_grouped:
            sigma2_j = variance.compute(center_data[outcome].values)
            print("center", center_id, sigma2_j)
            x_j = center_data[covariates].values
            y_j = center_data[outcome].values
            S_XX,S_YY = self.compute_SXX_SYY_for_center(x_j, y_j, sigma2, sigma2_j)
            return S_XX,S_YY

    def compute_SXX_SYY_for_center(self, X_j: np.array, Y_j: np.array, sigma2, sigma_u2):
        """Compute S_XX for a single center"""
        agg = NumpyAggregator(self.client)
        n_j=X_j.shape[0]
        n_j_total = agg.global_count(X_j)
        alpha_j = sigma_u2 / (sigma2 * (sigma2 + n_j_total * sigma_u2))
        #
        S_XX_1 = agg.fed_sum((1 / sigma2) * X_j.T @ X_j)
        _help = agg.fed_sum(X_j.T @  np.ones((n_j, 1)))
        S_XX_2 =  _help *_help.T *( - alpha_j )
        S_XX = S_XX_1 + S_XX_2
        #
        S_YY_1 = agg.fed_sum((1 / sigma2) * Y_j.T @ Y_j)
        _help = agg.fed_sum(Y_j.T @ np.ones((n_j, 1)))
        S_YY_2 = _help * _help.T * (- alpha_j)
        S_YY = S_YY_1 + S_YY_2
        S_YY = S_YY[0]
        return  S_XX, S_YY