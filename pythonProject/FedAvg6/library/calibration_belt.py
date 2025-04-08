__author__ = "Lucas Bulgarelli <lucas1@mit.edu>"
__version__ = "0.1.41"

import numpy as np
from tqdm import tqdm
from textwrap import dedent
import matplotlib.pyplot as plt

from scipy.stats import chi2
from scipy.special import logit, xlogy, expit
from scipy.integrate import quad as integrate
from scipy.optimize import brentq, minimize, NonlinearConstraint

import statsmodels.api as sm
import statsmodels.formula.api as smf

from math import sqrt, exp, pi, asin, atan

from data.numpy_dataset.np_fed_table import NumpyFedTable
from data.pandas.calibration_dataset import CalibrationDataset
from function.abstract_function import AggFunc


class CalibrationBelt(AggFunc):
    """Class for assessment of the calibration belt and goodness
    of fit of binomial models.

    Based on the work of Nattino et al.:

    Nattino, Giovanni, Stefano Finazzi, and Guido Bertolini.
    "A new calibration test and a reappraisal of the calibration
    belt for the assessment of prediction models based on dichotomous
    outcomes." Statistics in medicine 33.14 (2014): 2390-2407.
    """

    def __init__(self):
        super().__init__()
        self.p = None
        self.e = None
        self.n = None
        self.boundaries = None

    def compute(self, x: NumpyFedTable, y: NumpyFedTable):
        self.p: NumpyFedTable = x
        self.e: NumpyFedTable = y
        self.n = x.fed_count()
        self.boundaries = {}

    @classmethod
    def _cdf_m_1(cls, T):
        return chi2.cdf(T, 2)

    @classmethod
    def _cdf_m_2(cls, T, q):
        # Eq17
        ppf_q_1 = chi2.ppf(q, 1)
        cdf_t_1 = chi2.cdf(T, 1)
        return 1 / (1 - q) * (
            -2 * exp(-T / 2) / sqrt(2 * pi) *
            (sqrt(T) - sqrt(ppf_q_1)) +
            cdf_t_1 - q
        )

    @classmethod
    def _cdf_m_3(cls, T, q):
        # Eq18
        ppf_q_1 = chi2.ppf(q, 1)
        integrand = (
            lambda r:
            r *
            (exp((-r ** 2) / 2) - exp(-T / 2)) *
            (pi / 2 - 2 * asin(sqrt(ppf_q_1) / r))
        )
        lower = sqrt(2 * ppf_q_1)
        upper = sqrt(T)
        return (
            2 / (pi * ((1 - q) ** 2)) *
            integrate(integrand, lower, upper)[0]
        )

    @classmethod
    def _cdf_m_4(cls, T, q):
        # Eq19
        ppf_q_1 = chi2.ppf(q, 1)
        integrand = (
            lambda r:
            (r ** 2) *
            (exp((-r ** 2) / 2) - exp(-T / 2)) *
            (
                (-pi * sqrt(ppf_q_1) / (2 * r)) +
                2 * sqrt(ppf_q_1) / r *
                asin(((r ** 2) / ppf_q_1 - 1) ** (-1 / 2)) -
                2 * atan((1 - 2 * ppf_q_1 / (r ** 2)) ** (-1 / 2)) +
                2 * sqrt(ppf_q_1) / r * atan(
                    ((r ** 2) / ppf_q_1 - 2) ** (-1 / 2)) +
                2 * atan(r / sqrt(ppf_q_1) * sqrt((r ** 2) / ppf_q_1 - 2)) -
                2 * sqrt(ppf_q_1) / r * atan(sqrt((r ** 2) / ppf_q_1 - 2))
            )
        )
        lower = sqrt(3 * ppf_q_1)
        upper = sqrt(T)
        return (
            (2 / (pi * ((1 - q) ** 2))) ** (3 / 2) *
            integrate(integrand, lower, upper)[0]
        )

    @classmethod
    def calculate_cdf(cls, T, m, q=.95):
        if T <= (m - 1) * chi2.ppf(q, 1):
            return 0
        if m == 1:
            return cls._cdf_m_1(T)
        elif m == 2:
            return cls._cdf_m_2(T, q)
        elif m == 3:
            return cls._cdf_m_3(T, q)
        elif m == 4:
            return cls._cdf_m_4(T, q)
        else:
            return NotImplemented






    def forward_select(self, q, m_max=4, **kwargs):
        m_start = 1
        formula = "p ~ 1 + "
        inv_chi2 = chi2.ppf(q, 1)
        v= logit(self.e)
        data = {"p": self.p, "ge": logit(self.e)}

        for m1 in range(m_start, m_max+1):
            # Add new term
            formula += f"I(ge ** {m1})"

            if m1 >= m_start:
                # Fit logistic regression with current m
                family = sm.families.Binomial()
                model1 = smf.glm(formula=formula, data=data,
                                 family=family).fit()

                if m1 > m_start:
                    # Log-likelihood ratio test (Eq6)
                    Dm = 2 * (model1.llf - model.llf)

                    # If the model doesn't improve than
                    # we use previo
                    # us order as optimal
                    # pchisq(fit$deviance - fitNew$deviance, 1) < thres
                    if Dm < inv_chi2:
                        m = m1 - 1
                        break

                # Retain log-likelihood to
                # compare in next iteration
                m = m1
                model = model1

            formula += " + "

        return m, model

    def test(self, q=.95, **kwargs):
        # Forward select (m)
        m, model = self.forward_select(q, **kwargs)

        # Compute stat (Eq9)
        llh =(xlogy(self.p, self.e) + xlogy(1 - self.p, 1 - self.e)).fed_sum()
        T = 2 * (model.llf - llh)
        p_value = 1 - self.calculate_cdf(T, m, q)

        return T, p_value

    def _root_fun(self, x, *args):
        m, q, confidence = args
        return self.calculate_cdf(x, m, q) - confidence

    @staticmethod
    def _fun(alpha, *args):
        # Eq26
        geM, sign = args
        return sign * (alpha @ geM)

    @staticmethod
    def _jac(alpha, *args):
        # Eq29
        geM, sign = args
        return sign * geM

    def calculate_boundaries(self, confidence, size=50, q=.95, **kwargs):
        # Forward select parameter m
        m, model = self.forward_select(q, **kwargs)

        # We will cache boundaries for confidence intervals
        # so we save the parameters used in their calculation.
        if confidence in self.boundaries:
            params = self.boundaries[confidence]["params"]
            # If a request for the same interval is requested
            # it is only calculted one of the parameters change.
            # The parameters that modify the belt are (size, q, m).
            # In the case `size` changes, we compute the boundaries
            # only if the new `size` is greater than the previously used.
            if (size <= params["size"] and
                    q == params["q"] and
                    m == params["m"]):
                return self.boundaries[confidence]["boundaries"]

        # New parameters
        params = {"size": size, "q": q, "m": m}

        # Find ky
        a, b = (m - 1) * chi2.ppf(q, 1), 40
        args = m, q, confidence
        k = brentq(self._root_fun, a, b, args=args)

        # Calculate logit(E) matrix
        M = np.linspace([0], [m], num=m+1, axis=1)
        Ge = logit(self.e)[np.newaxis]
        GeM = Ge.T ** M

        # Upper boundary (Eq27)
        boundary = model.llf - k / 2

        # Create subset based on size
        logit_sub = np.linspace(np.min(Ge), np.max(Ge), num=size//2)
        e_sub = np.linspace(np.min(self.e), np.max(self.e), num=size // 2)
        Ge_sub = np.sort(np.append(logit_sub, logit(e_sub)))
        GeM_sub = Ge_sub[np.newaxis].T ** M

        # Constraint function (Eq27)
        def fun_lalpha(alpha):
            # Calculate probability
            alphaE = expit(GeM @ alpha)

            # Clip probability to epsilon so
            # we can compute log-likelihood
            eps = 1e-5
            alphaE = np.clip(alphaE, eps, 1-eps)

            # Compute Log-likelihood
            lalpha = xlogy(self.p, alphaE) + xlogy(1 - self.p, 1 - alphaE)
            return lalpha.fed_sum()

        def jac_lalpha(alpha):
            # Calculate probability
            alphaE = expit(GeM @ alpha)
            return (self.p - alphaE) @ GeM

        lower, upper = [], []
        for geM in tqdm(GeM_sub):
            constraints = NonlinearConstraint(
                fun_lalpha,
                boundary, 0,
                jac_lalpha,
                keep_feasible=True
            )

            # Minimize alpha to find lower bound
            args = (geM, 1)
            min_alpha = minimize(
                fun=self._fun, x0=model.params, args=args,
                method='trust-constr', jac=self._jac,
                hess=lambda alpha, *args: np.zeros((m+1,)),
                constraints=constraints, tol=1e-5
            ).x

            # Maximize alpha to find upper bound
            args = (geM, -1)
            max_alpha = minimize(
                fun=self._fun, x0=model.params, args=args,
                method='trust-constr', jac=self._jac,
                hess=lambda alpha, *args: np.zeros((m+1,)),
                constraints=constraints, tol=1e-5
            ).x

            # Calculate bounds
            lower.append(expit(min_alpha @ geM))
            upper.append(expit(max_alpha @ geM))

        # Save parameters
        boundaries = np.array([expit(Ge_sub), lower, upper]).T
        self.boundaries[confidence] = {
            "params": params,
            "boundaries": boundaries
        }
        return boundaries

    def plot(self, confidences=None, q=.95, **kwargs):
        if confidences is None:
            confidences = [.8, .95]

        # Select value of m
        m, _ = self.forward_select(q, **kwargs)

        # Calculate p-value
        _, p_value = self.test(q, **kwargs)
        if p_value < .001:
            p_text = "< 0.001"
        else:
            p_text = f"{p_value:.3f}"

        # Calculate boundaries for each confidence level
        for confidence in confidences:
            self.calculate_boundaries(confidence, q=q, **kwargs)

        # Plot stats
        fig, ax = plt.subplots(1, figsize=[15, 12])
        plt.text(
            0, 1.035,
            dedent(f"""
            Polynomial degree: {m}
            p-value: {p_text}
            n: {self.n}
            """),
            size=18, va='top'
        )

        # Plot perfect calibration doted line
        ax.plot([0, 1], [0, 1], linestyle='--', color='k')

        # Plot belt for each confidence level
        confidences.sort(reverse=True)
        viridis = plt.cm.get_cmap("viridis")
        for i, confidence in enumerate(confidences):
            alpha = .9 / len(confidences)
            [Ge, lower, upper] = self.boundaries[confidence]["boundaries"].T
            ax.fill_between(Ge, lower, upper, color=viridis(i), alpha=alpha)

        return fig, ax



def main():
    dataset = CalibrationDataset(0, 1)

    # Separate P (outcome) and E (probability outputed by the model)
    # In this case we will evaluate two different models
    p = dataset.get_attribute('target')
    predictors = {
        'SVM': dataset.get_attribute('SVM'),
        'NN': dataset.get_attribute('NN')
    }
    belts = {}
    for key, e in predictors.items():
        belts[key] = CalibrationBelt()
        belts[key].compute(p,e)
        print(belts)
    for model, belt in belts.items():
        fig, ax  = belt.plot(confidences=[.8, .95])
        ax.set_title(model, fontsize=30)
    for model, belt in belts.items():
        T, p_value = belt.test()
        print(f"Model: {model:3}, T-statistic: {T:08.5f} , p-value: {p_value:07.5f}")
        boundaries = belt.calculate_boundaries(.95)
        lower, upper = boundaries[0, 1:]
        print(f"Lower bound: {lower:.4f}, Upper bound: {upper:.4f}")
        T = [i / 10 for i in range(0, 301)]
        viridis = plt.cm.get_cmap("viridis", 4)
        plt.show()
        fig, ax = plt.subplots(1, figsize=[10, 7])
        for m in [1, 2, 3, 4]:
            cdf = []
            for t in T:
                cdf.append(CalibrationBelt.calculate_cdf(t, m, .8))
            ax.plot(T, cdf, color=viridis(m-1))
        plt.show()

# main()