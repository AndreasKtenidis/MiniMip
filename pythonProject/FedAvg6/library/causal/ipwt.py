from library.stat_models.fed_glm import Fed_GLM
from library.templates.statistical_function import StatisticalFunction
import pandas as pd

class IPWT(StatisticalFunction):
    '''IPTW adjusts for confounding by making the treatment and control groups comparable.'''

    def compute(self, data:pd.DataFrame,*, treatment:str, outcome:str, confounders:list[str]):
        '''IPTW adjusts for confounding by making the treatment and control groups comparable.
            treatment: Treatment indicator (binary: treated vs.untreated)
            outcome: Outcome y variable
            confounders: Covariates / confounders(e.g., age, gender, severity)
        '''
        all_attributes = confounders + [treatment, outcome]
        data=data[all_attributes].dropna()

        # **********************************************
        _coufounders=data[confounders].values
        _treatment=data[[treatment]].values
        _outcome = data[[outcome]].values
        # **********************************************
        agg = self.get_numpy_aggregator()
        glm = Fed_GLM(self.client)
        glm.fit(_coufounders, _treatment)
        # Predict propensity scores
        propensity = glm.predict(_coufounders)

        # predict weights
        weights = _treatment / propensity + (1 - _treatment) / (1 - propensity)

        treated_mean = agg.global_sum(_outcome * weights * _treatment) / agg.global_sum((weights * _treatment))
        untreated_mean = agg.global_sum((_outcome * weights * (1 - _treatment))) / agg.global_sum(
            (weights * (1 - _treatment)))

        treatment_effect = treated_mean - untreated_mean
        return treatment_effect
