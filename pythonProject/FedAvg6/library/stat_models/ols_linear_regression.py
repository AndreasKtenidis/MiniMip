import numpy as np

from library.stat_models._statistical_Model import StatisticalModel

class FedOLS(StatisticalModel):
    """Ordinary Least Squares linear regression."""

    # def compute(self, x:np.ndarray, y:np.ndarray):
    #     self.fit(x,y)
    #     for i in range(len(x)):
    #         out = self.predict(x[i])
    #         print(y[i],'vs',out)


    def fit(self,x:np.ndarray, y:np.ndarray):
        xtx = self.aggregator.fed_sum(x.T @ x)
        xty = self.aggregator.fed_sum(x.T @ y)
        self.b_dot = np.linalg.inv(xtx) @ xty

    def predict(self,x:np.ndarray):
        return np.sum(self.b_dot * x)

    def __init__(self, client):
        super().__init__(client)
        self.aggregator = self.get_numpy_aggregator()
        self.b_dot=None


