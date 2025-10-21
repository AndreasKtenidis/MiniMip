from library.utils.numpy_aggregator import NumpyAggregator
from library.under_development.stat_models.logistic_regression_saga_solver import FederatedLogisticRegressionClientSaSo
from library.core.statistical_function import StatisticalFunction
import pandas as pd

class IPWT(StatisticalFunction):
    '''IPTW adjusts for confounding by making the treatment and control groups comparable.'''

    def compute(self, data:pd.DataFrame,*, treatment:str, confounders:list[str]):
        '''IPTW adjusts for confounding by making the treatment and control groups comparable.
            treatment: Treatment indicator (binary: treated vs.untreated)
            outcome: Outcome y variable
            confounders: Covariates / confounders(e.g., age, gender, severity)
        '''
        all_attributes = confounders + [treatment]
        data=data[all_attributes].dropna()
        # **********************************************
        _coufounders=data[confounders].values
        _treatment=data[treatment].values
        # **********************************************
        agg = NumpyAggregator(self.client)
        model = FederatedLogisticRegressionClientSaSo(self.client)
        model.fit(_coufounders, _treatment)
        # Predict propensity scores
        propensity = model.predict_proba(_coufounders)[:, 1]
        data['ps'] = propensity

        weights = _treatment / propensity + (1 - _treatment) / (1 - propensity)
        data['weight']= weights
        return data
