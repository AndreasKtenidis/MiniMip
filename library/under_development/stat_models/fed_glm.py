from statsmodels.api import GLM, families

from library.utils.aggregation_client import AggregationClientInterface
from library.utils.numpy_aggregator import NumpyAggregator
from library.core.statistical_model import StatisticalModel
import numpy as np


class Fed_GLM(StatisticalModel):

    def __init__(self, client: AggregationClientInterface):
        super().__init__(client)
        self.params = None
        self.xtwx = None
        self.llf = None
        self.model = None
        self.aggregator = NumpyAggregator(self.client)

    def fit(self, x, y, max_iter=50):
        model = GLM(y, x, family=families.Binomial())
        result = model.fit(disp=0)
        self._update(model, result)
        # pred_probs = model.predict()  # Predicted probabilities
        # weights = pred_probs * (1 - pred_probs)  # W = diag(p*(1-p))

        # aggregator.global_sum(result.llf)

        # aggregator.fed_sum(np.dot(input.T, input * weights[:, np.newaxis]))

        # aggregator.fed_avg(result.params)
        return self

    def predict(self, x, which=None):
        """Predict probabilities using federated parameters."""
        linear_pred = np.dot(x, self.params)
        # return 1 / (1 + np.exp(-linear_pred))
        y = None
        if which == "linear":
            return linear_pred
        elif which is None:
            return 1 / (1 + np.exp(-linear_pred))

        # return y

    def _cov_params(self):
        return self._cov_params

    def _update(self, model, result):
        # Adjust the parameters of the model

        # 1. Update log-likelihood
        self.llf = self.aggregator.fed_sum(result.llf)

        # 2. Update parameters (Fisher-weighted)
        xtwx = model.exog.T @ np.diag(model.weights) @ model.exog

        self.params = np.linalg.solve(
            self.aggregator.fed_sum(xtwx),
            self.aggregator.fed_sum(xtwx @ result.params)
        )

        # 3. Update covariance
        n = len(model.endog)
        if isinstance(model.family, (families.Binomial, families.Poisson)):
            combined_scale = 1  # No dispersion
        else:
            combined_scale = self.aggregator.fed_sum(model.scale * n) / self.aggregator.fed_sum(n)
        self._cov_params = np.linalg.inv(self.aggregator.fed_sum(xtwx) * combined_scale)
