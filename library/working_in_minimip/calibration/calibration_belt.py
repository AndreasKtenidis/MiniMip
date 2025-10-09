import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from statsmodels.tools import add_constant

from library.templates.statistical_function import StatisticalFunction
from library.working_in_minimip.stat_models.fed_glm import Fed_GLM


class CalibrationBelt(StatisticalFunction):

    def compute(self, e, o, confidence=0.99, max_poly_degree=5):
        """
        Compute and plot the calibration belt for predicted (e) vs. observed (o) binary outcomes.

        Args:
            e (array-like): Expected probabilities (0 < e < 1).
            o (array-like): Observed binary outcomes (0 or 1).
            confidence (float): Confidence level (e.g., 0.95 for 95% bands).
            max_poly_degree (int): Maximum polynomial degree for the logit calibration curve.

        Returns:
            dict: Fitted model parameters and calibration belt data.
        """
        # Input validation
        e = np.clip(np.asarray(e), 1e-6, 1 - 1e-6)  # Avoid logit(0) or logit(1)
        o = np.asarray(o)

        # Step 1: Transform expected probabilities to logits (g_e)
        g_e = np.log(e / (1 - e))
        # Step 2: Fit polynomial logistic regression (up to max_poly_degree)
        best_m = 1
        best_model = None
        best_ll = -np.inf

        for m in range(1, max_poly_degree + 1):
            # Design matrix: [1, g_e, g_e^2, ..., g_e^m]
            x_vec = np.column_stack([g_e ** i for i in range(m + 1)])
            model = Fed_GLM(self.client)
            model.fit(x=add_constant(x_vec), y=o)

            # Likelihood-ratio test (compare to previous model)
            if m > 1:
                lr_stat = 2 * (model.llf - best_ll)
                p_value = chi2.sf(lr_stat, df=1)
                if p_value > 0.01:  # Stop if improvement is insignificant (q=0.99)
                    break

            best_m = m
            best_model = model
            best_ll = model.llf

        # Step 3: Compute calibration curve
        agg = self.get_numpy_aggregator()
        g_e_range = np.linspace(agg.global_min(g_e), agg.global_max(g_e), 50)
        x_range = np.column_stack([g_e_range ** i for i in range(best_m + 1)])
        p_pred = best_model.predict(add_constant(x_range))

        # Step 4: Compute confidence band
        cov_matrix = best_model._cov_params()
        se = np.sqrt(np.sum([x_range[:, i] * x_range[:, j] * cov_matrix[i, j]
                             for i in range(best_m + 1) for j in range(best_m + 1)], axis=0))
        chi2_val = chi2.ppf(confidence, df=2)
        g_p_lower = best_model.predict(add_constant(x_range), which="linear") - np.sqrt(chi2_val) * se
        g_p_upper = best_model.predict(add_constant(x_range), which="linear") + np.sqrt(chi2_val) * se
        # Back-transform to probabilities
        p_lower = 1 / (1 + np.exp(-g_p_lower))
        p_upper = 1 / (1 + np.exp(-g_p_upper))
        e_range = 1 / (1 + np.exp(-g_e_range))  # Back to e-scale for plotting

        # Step 5: Plot
        plt.figure(figsize=(10, 6))
        plt.plot(e_range, p_pred, label=f'Calibration curve (m={best_m})', color='blue')
        plt.fill_between(e_range, p_lower, p_upper, alpha=0.2, color='gray',
                         label=f'{int(confidence * 100)}% Calibration belt')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.xlabel('Expected Probability (e)')
        plt.ylabel('Observed Probability (p)')
        plt.title('Calibration Belt')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
