
from statsmodels.api import GLM, families

from client.aggregation_client import AggregationClient
from function.abstract_function import AggFunc
import numpy as np

class Fed_GLM(AggFunc):

    def __init__(self,client:AggregationClient):
        super().__init__(client)
        self.params = None
        self.xtwx = None
        self.llf = None

    def train(self, input, output, max_iter=50):
        aggregator = self.get_numpy_aggregator()
        model = GLM(output, input, family=families.Binomial())
        result = model.fit()

        pred_probs = result.predict()  # Predicted probabilities
        weights = pred_probs * (1 - pred_probs)  # W = diag(p*(1-p))
        self.llf = aggregator.global_sum(result.llf)
        self.xtwx = aggregator.fed_sum(np.dot(input.T, input * weights[:, np.newaxis]))
        self.params = aggregator.fed_avg(result.params)
        print(self.llf)


    def predict(self, x):
        """Predict probabilities using federated parameters."""
        linear_pred = np.dot(x, self.params)
        return 1 / (1 + np.exp(-linear_pred))

    def cov_params(self):
        return self.xtwx