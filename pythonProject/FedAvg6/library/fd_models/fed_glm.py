
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
        self.model=None

    def train(self, input, output, max_iter=50):
        aggregator = self.get_numpy_aggregator()

        model = GLM(output, input, family=families.Binomial()).fit(disp=0)
        self.update(model)
        # pred_probs = model.predict()  # Predicted probabilities
        # weights = pred_probs * (1 - pred_probs)  # W = diag(p*(1-p))

        #aggregator.global_sum(result.llf)

        #aggregator.fed_sum(np.dot(input.T, input * weights[:, np.newaxis]))

        # aggregator.fed_avg(result.params)
        return self


    def predict(self, x,which=None):
        """Predict probabilities using federated parameters."""
        # linear_pred = np.dot(x, self.params)
        # return 1 / (1 + np.exp(-linear_pred))
        y=None
        if which == "linear":
            y= self.model.predict(x, which="linear")
        elif which is None:
            y = self.model.predict(x)
        self.update(self.model)
        return y

    def cov_params(self):
        return self.model.cov_params()

    def update(self,model):
        self.llf = model.llf
        self.xtwx = model.cov_params()
        self.params = model.params
        self.model = model