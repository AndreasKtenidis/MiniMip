from library.stat_models.fed_glm import Fed_GLM
from library.templates.statistical_function import StatisticalFunction


class IPWT(StatisticalFunction):
    '''IPTW adjusts for confounding by making the treatment and control groups comparable.'''

    def compute(self, treatment, outcome, confounders):
        '''IPTW adjusts for confounding by making the treatment and control groups comparable.
            treatment: Treatment indicator (binary: treated vs.untreated)
            outcome: Outcome y variable
            confounders: Covariates / confounders(e.g., age, gender, severity)
        '''
        agg = self.get_numpy_aggregator()

        glm = Fed_GLM(self.client)
        glm.fit(confounders,treatment)

        # Predict propensity scores
        propensity = glm.predict(confounders)

        # predict weights
        weights = treatment/propensity+(1-treatment)/(1-propensity)

        treated_mean = agg.global_sum(outcome * weights*treatment) / agg.global_sum((weights*treatment))
        untreated_mean = agg.global_sum((outcome * weights * (1-treatment))) / agg.global_sum((weights * (1-treatment)))

        treatment_effect = treated_mean - untreated_mean
        return treatment_effect

