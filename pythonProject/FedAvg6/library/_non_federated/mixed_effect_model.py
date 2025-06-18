from __future__ import annotations

from linearmodels.compat.formulaic import FORMULAIC_GTE_0_6, future_ordering

from typing import Any, Mapping, NamedTuple, Type, Union, cast

from formulaic.formula import Formula
from formulaic.model_spec import NAAction
from formulaic.parser.algos.tokenize import tokenize
from formulaic.utils.context import capture_context
import numpy as np
from pandas import Categorical, DataFrame, Index, MultiIndex, Series
from scipy.linalg import lstsq as sp_lstsq


from linearmodels.panel.covariance import (
    ACCovariance,
    ClusteredCovariance,
    CovarianceManager,
    DriscollKraay,
    HeteroskedasticCovariance,
    HomoskedasticCovariance,
    setup_covariance_estimator,
)
from linearmodels.panel.data import PanelData, PanelDataLike
from linearmodels.panel.results import (
    RandomEffectsResults,
)

from linearmodels.shared.exceptions import (
    missing_warning,
)
from linearmodels.shared.hypotheses import (
    InapplicableTestStatistic,
    InvalidTestStatistic,
    WaldTestStatistic,
)
from linearmodels.shared.linalg import has_constant
from linearmodels.shared.typed_getters import get_panel_data_like
from linearmodels.shared.utility import AttrDict
from linearmodels.typing import (
    ArrayLike,
    BoolArray,
    Float64Array,
    IntArray,
)

CovarianceEstimator = Union[
    ACCovariance,
    ClusteredCovariance,
    DriscollKraay,
    HeteroskedasticCovariance,
    HomoskedasticCovariance,
]

CovarianceEstimatorType = Union[
    Type[ACCovariance],
    Type[ClusteredCovariance],
    Type[DriscollKraay],
    Type[HeteroskedasticCovariance],
    Type[HomoskedasticCovariance],
]


def _lstsq(
    x: Float64Array, y: Float64Array, rcond: float | None = None
) -> tuple[Float64Array, Float64Array, int, Float64Array]:
    if rcond is None:
        eps = np.finfo(np.float64).eps
        cond = float(max(x.shape) * eps)
    else:
        cond = rcond
    return sp_lstsq(x, y, cond=cond, lapack_driver="gelsy")


def panel_structure_stats(ids: IntArray, name: str) -> Series:
    bc = np.bincount(ids)
    index = ["mean", "median", "max", "min", "total"]
    out = [bc.mean(), np.median(bc), bc.max(), bc.min(), bc.shape[0]]
    return Series(out, index=index, name=name)


class FInfo(NamedTuple):
    sel: BoolArray
    name: str
    invalid_test_stat: InvalidTestStatistic | None
    is_invalid: bool


def _deferred_f(
    params: Series, cov: DataFrame, debiased: bool, df_resid: int, f_info: FInfo
) -> InvalidTestStatistic | WaldTestStatistic:
    if f_info.is_invalid:
        assert f_info.invalid_test_stat is not None
        return f_info.invalid_test_stat
    sel = f_info.sel
    name = f_info.name
    test_params = np.asarray(params)[sel]
    test_cov = np.asarray(cov)[sel][:, sel]
    test_stat = test_params.T @ np.linalg.inv(test_cov) @ test_params
    test_stat = float(test_stat)
    df = int(sel.sum())
    null = "All parameters ex. constant are zero"

    if debiased:
        wald = WaldTestStatistic(test_stat / df, null, df, df_resid, name=name)
    else:
        wald = WaldTestStatistic(test_stat, null, df, name=name)
    return wald


class PanelFormulaParser:
    """
    Parse formulas for OLS and IV models

    Parameters
    ----------
    formula : str
        String formula object.
    data : DataFrame
        Frame containing values for variables used in formula
    context : int
        Stack depth to use when evaluating formulas

    Notes
    -----
    The general structure of a formula is `dep ~ exog`
    """

    def __init__(
        self,
        formula: str,
        data: PanelDataLike,
        eval_env: int = 2,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self._formula = formula
        self._data = PanelData(data, convert_dummies=False, copy=False)
        self._na_action = NAAction("ignore")
        self._eval_env = eval_env
        if not context:
            self._context = capture_context(eval_env)
        else:
            self._context = context
        self._dependent = self._exog = None
        if FORMULAIC_GTE_0_6 and not future_ordering():
            self._formulaic_kwargs = dict(_ordering="sort")
        else:
            self._formulaic_kwargs = {}
        self._parse()

    def _parse(self) -> None:
        parts = self._formula.split("~")
        parts[1] = " 0 + " + parts[1]
        cln_formula = "~".join(parts)
        rm_list = []
        effects = {"EntityEffects": False, "FixedEffects": False, "TimeEffects": False}
        tokens = tokenize(cln_formula)
        for token in tokens:
            _token = str(token)
            if _token in effects:
                effects[_token] = True
                rm_list.append(_token)
        if effects["EntityEffects"] and effects["FixedEffects"]:
            raise ValueError("Cannot use both FixedEffects and EntityEffects")
        self._entity_effect = effects["EntityEffects"] or effects["FixedEffects"]
        self._time_effect = effects["TimeEffects"]
        for effect in effects:
            if effects[effect]:
                loc = cln_formula.find(effect)
                start = cln_formula.rfind("+", 0, loc)
                end = loc + len(effect)
                cln_formula = cln_formula[:start] + cln_formula[end:]
        self._lhs, self._rhs = map(lambda s: "0 + " + s.strip(), cln_formula.split("~"))

    @property
    def entity_effect(self) -> bool:
        """Formula contains entity effect"""
        return self._entity_effect

    @property
    def time_effect(self) -> bool:
        """Formula contains time effect"""
        return self._time_effect

    @property
    def eval_env(self) -> int:
        """Set or get the eval env depth"""
        return self._eval_env

    @eval_env.setter
    def eval_env(self, value: int) -> None:
        self._context = capture_context(value)
        self._eval_env = value

    @property
    def context(self) -> Mapping[str, Any]:
        """Get the context used in the parser"""
        return self._context

    @context.setter
    def context(self, value: Mapping[str, Any]) -> None:
        """Get the context used in the parser"""
        self._context = value

    @property
    def data(self) -> tuple[DataFrame, DataFrame]:
        """Returns a tuple containing the dependent, exog, endog"""
        out = self.dependent, self.exog
        return out

    @property
    def dependent(self) -> DataFrame:
        """DataFrame containing the dependent variable"""
        return DataFrame(
            Formula(self._lhs, **self._formulaic_kwargs).get_model_matrix(
                self._data.dataframe,
                context=self._context,
                na_action=self._na_action,
            )
        )

    @property
    def exog(self) -> DataFrame:
        """DataFrame containing the exogenous variables"""
        return DataFrame(
            Formula(self._rhs, **self._formulaic_kwargs).get_model_matrix(
                self._data.dataframe,
                context=self._context,
                na_action=self._na_action,
            )
        )


class AmbiguityError(Exception):
    pass


__all__ = [
    "RandomEffects",
]


# Likely
# TODO: Formal test of other outputs
# Future
# TODO: Bootstrap covariance
# TODO: Possibly add AIC/BIC
# TODO: ML Estimation of RE model
# TODO: Defer estimation of 3 R2 values -- slow

EXOG_PREDICT_MSG = """\
exog does not have the correct number of columns. Saw {x_shape}, expected
{params_shape}. This can happen since exog is converted to a PanelData object
before computing the fitted value. The best practice is to pass a DataFrame
with a 2-level MultiIndex containing the entity- and time-ids."""


class _PanelModelBase:
    r"""
    Base class for all panel models

    Parameters
    ----------
    dependent : array_like
        Dependent (left-hand-side) variable (time by entity)
    exog : array_like
        Exogenous or right-hand-side variables (variable by time by entity).
    weights : array_like
        Weights to use in estimation.  Assumes residual variance is
        proportional to inverse of weight to that the residual time
        the weight should be homoskedastic.
    check_rank : bool
        Flag indicating whether to perform a rank check on the exogenous
        variables to ensure that the model is identified. Skipping this
        check can reduce the time required to validate a model specification.
        Results may be numerically unstable if this check is skipped and
        the matrix is not full rank.
    """

    def __init__(
        self,
        dependent: PanelDataLike,
        exog: PanelDataLike,
        *,
        weights: PanelDataLike | None = None,
        check_rank: bool = True,
    ) -> None:
        self.dependent = PanelData(dependent, "Dep")
        self.exog = PanelData(exog, "Exog")
        self._original_shape = self.dependent.shape
        self._constant = False
        self._formula: str | None = None
        self._is_weighted = True
        self._name = self.__class__.__name__
        self.weights = self._adapt_weights(weights)
        self._not_null = np.ones(self.dependent.values2d.shape[0], dtype=bool)
        self._cov_estimators = CovarianceManager(
            self.__class__.__name__,
            HomoskedasticCovariance,
            HeteroskedasticCovariance,
            ClusteredCovariance,
            DriscollKraay,
            ACCovariance,
        )
        self._original_index = self.dependent.index.copy()
        self._constant_index: int | None = None
        self._check_rank = bool(check_rank)
        self._validate_data()
        self._singleton_index: BoolArray | None = None

    def __str__(self) -> str:
        out = "{name} \nNum exog: {num_exog}, Constant: {has_constant}"
        return out.format(
            name=self.__class__.__name__,
            num_exog=self.exog.dataframe.shape[1],
            has_constant=self.has_constant,
        )

    def __repr__(self) -> str:
        return self.__str__() + "\nid: " + str(hex(id(self)))

    def reformat_clusters(self, clusters: IntArray | PanelDataLike) -> PanelData:
        """
        Reformat cluster variables

        Parameters
        ----------
        clusters : array_like
            Values to use for variance clustering

        Returns
        -------
        PanelData
            Original data with matching axis and observation dropped where
            missing in the model data.

        Notes
        -----
        This is exposed for testing and is not normally needed for estimation
        """
        clusters_pd = PanelData(clusters, var_name="cov.cluster", convert_dummies=False)
        if clusters_pd.shape[1:] != self._original_shape[1:]:
            raise ValueError(
                "clusters must have the same number of entities "
                "and time periods as the model data."
            )
        clusters_pd.drop(~self.not_null)
        return clusters_pd.copy()

    def _info(self) -> tuple[Series, Series, DataFrame | None]:
        """Information about panel structure"""

        entity_info = panel_structure_stats(
            self.dependent.entity_ids.squeeze(), "Observations per entity"
        )
        time_info = panel_structure_stats(
            self.dependent.time_ids.squeeze(), "Observations per time period"
        )
        other_info = None

        return entity_info, time_info, other_info

    def _adapt_weights(self, weights: PanelDataLike | None) -> PanelData:
        """Check and transform weights depending on size"""
        if weights is None:
            self._is_weighted = False
            frame = self.dependent.dataframe.copy()
            frame.iloc[:, :] = 1
            # TODO: Remove once pandas typing fixed
            frame.columns = Index(["weight"])
            return PanelData(frame)

        frame = DataFrame(columns=self.dependent.entities, index=self.dependent.time)
        nobs, nentity = self.exog.nobs, self.exog.nentity

        if weights.ndim == 3 or weights.shape == (nobs, nentity):
            return PanelData(weights)

        if isinstance(weights, np.ndarray):
            weights = cast(Float64Array, np.squeeze(weights))
        if weights.shape[0] == nobs and nobs == nentity:
            raise AmbiguityError(
                "Unable to distinguish nobs form nentity since they are "
                "equal. You must use an 2-d array to avoid ambiguity."
            )
        if (
            isinstance(weights, (Series, DataFrame))
            and isinstance(weights.index, MultiIndex)
            and weights.shape[0] == self.dependent.dataframe.shape[0]
        ):
            frame = DataFrame(weights)
        elif weights.shape[0] == nobs:
            weights_arr = np.asarray(weights)[:, None]
            weights_arr = weights_arr @ np.ones((1, nentity))

            frame.iloc[:, :] = weights_arr
        elif weights.shape[0] == nentity:
            weights_arr = np.asarray(weights)[None, :]
            weights_arr = np.ones((nobs, 1)) @ weights_arr
            frame.iloc[:, :] = weights_arr
        elif weights.shape[0] == nentity * nobs:
            frame = self.dependent.dataframe.copy()
            frame.iloc[:, :] = np.asarray(weights)[:, None]
        else:
            raise ValueError("Weights do not have a supported shape.")
        return PanelData(frame)

    def _check_exog_rank(self) -> int:
        if not self._check_rank:
            return self.exog.shape[1]
        x = cast(Float64Array, self.exog.values2d)
        _, _, rank_of_x, _ = _lstsq(x, np.ones(x.shape[0]))
        if rank_of_x < x.shape[1]:
            raise ValueError(
                "exog does not have full column rank. If you wish to proceed with "
                "model estimation irrespective of the numerical accuracy of "
                "coefficient estimates, you can set check_rank=False."
            )
        return rank_of_x

    def _validate_data(self) -> None:
        """Check input shape and remove missing"""
        y = self._y = cast(Float64Array, self.dependent.values2d)
        x = self._x = cast(Float64Array, self.exog.values2d)
        w = self._w = cast(Float64Array, self.weights.values2d)
        if y.shape[0] != x.shape[0]:
            raise ValueError(
                "dependent and exog must have the same number of "
                "observations. The number of observations in dependent "
                f"is {y.shape[0]}, and the number of observations in exog "
                f"is {x.shape[0]}."
            )
        if y.shape[0] != w.shape[0]:
            raise ValueError(
                "weights must have the same number of " "observations as dependent."
            )

        all_missing = np.any(np.isnan(y), axis=1) & np.all(np.isnan(x), axis=1)
        missing = (
            np.any(np.isnan(y), axis=1)
            | np.any(np.isnan(x), axis=1)
            | np.any(np.isnan(w), axis=1)
        )

        missing_warning(np.asarray(all_missing ^ missing), stacklevel=4)
        if np.any(missing):
            self.dependent.drop(missing)
            self.exog.drop(missing)
            self.weights.drop(missing)

            x = cast(Float64Array, self.exog.values2d)
            self._not_null = np.asarray(~missing)

        w_df = self.weights.dataframe
        if np.any(np.asarray(w_df) <= 0):
            raise ValueError("weights must be strictly positive.")
        w_df = w_df / w_df.mean()
        self.weights = PanelData(w_df)
        rank_of_x = self._check_exog_rank()
        self._constant, self._constant_index = has_constant(x, rank_of_x)

    @property
    def formula(self) -> str | None:
        """Formula used to construct the model"""
        return self._formula

    @formula.setter
    def formula(self, value: str | None) -> None:
        self._formula = value

    @property
    def has_constant(self) -> bool:
        """Flag indicating the model a constant or implicit constant"""
        return self._constant

    def _f_statistic(
        self,
        weps: Float64Array,
        y: Float64Array,
        x: Float64Array,
        root_w: Float64Array,
        df_resid: int,
    ) -> WaldTestStatistic | InvalidTestStatistic:
        """Compute model F-statistic"""
        weps_const = y
        num_df = x.shape[1]
        name = "Model F-statistic (homoskedastic)"
        if self.has_constant:
            if num_df == 1:
                return InvalidTestStatistic("Model contains only a constant", name=name)

            num_df -= 1
            weps_const = cast(
                Float64Array,
                y - float(np.squeeze((root_w.T @ y) / (root_w.T @ root_w))),
            )

        resid_ss = float(np.squeeze(weps.T @ weps))
        num = float(np.squeeze(weps_const.T @ weps_const - resid_ss))
        denom = resid_ss
        denom_df = df_resid
        stat = float((num / num_df) / (denom / denom_df)) if denom > 0.0 else 0.0
        return WaldTestStatistic(
            stat,
            null="All parameters ex. constant are zero",
            df=num_df,
            df_denom=denom_df,
            name=name,
        )

    def _f_statistic_robust(
        self,
        params: Float64Array,
    ) -> FInfo:
        """Compute Wald test that all parameters are 0, ex. constant"""
        sel = np.ones(params.shape[0], dtype=bool)
        name = "Model F-statistic (robust)"

        if self.has_constant:
            if len(sel) == 1:
                return FInfo(
                    sel,
                    name,
                    InvalidTestStatistic("Model contains only a constant", name=name),
                    True,
                )
            assert isinstance(self._constant_index, int)
            sel[self._constant_index] = False

        return FInfo(sel, name, None, False)

    def _prepare_between(self) -> tuple[Float64Array, Float64Array, Float64Array]:
        """Prepare values for between estimation of R2"""
        weights = self.weights if self._is_weighted else None
        y = np.asarray(self.dependent.mean("entity", weights=weights))
        x = np.asarray(self.exog.mean("entity", weights=weights))
        # Weight transformation
        wcount, wmean = self.weights.count("entity"), self.weights.mean("entity")
        wsum = wcount * wmean
        w = np.asarray(wsum)
        w = w / w.mean()

        return y, x, w

    def _rsquared_corr(self, params: Float64Array) -> tuple[float, float, float]:
        """Correlation-based measures of R2"""
        # Overall
        y = self.dependent.values2d
        x = self.exog.values2d
        xb = x @ params
        r2o = 0.0
        if y.std() > 0 and xb.std() > 0:
            r2o = np.corrcoef(y.T, (x @ params).T)[0, 1]
        # Between
        y = np.asarray(self.dependent.mean("entity"))
        x = np.asarray(self.exog.mean("entity"))
        xb = x @ params
        r2b = 0.0
        if y.std() > 0 and xb.std() > 0:
            r2b = np.corrcoef(y.T, (x @ params).T)[0, 1]

        # Within
        y = self.dependent.demean("entity", return_panel=False)
        x = self.exog.demean("entity", return_panel=False)
        xb = x @ params
        r2w = 0.0
        if y.std() > 0 and xb.std() > 0:
            r2w = np.corrcoef(y.T, xb.T)[0, 1]

        return r2o**2, r2w**2, r2b**2

    def _rsquared(
        self, params: Float64Array, reweight: bool = False
    ) -> tuple[float, float, float]:
        """Compute alternative measures of R2"""
        if self.has_constant and self.exog.nvar == 1:
            # Constant only fast track
            return 0.0, 0.0, 0.0

        #############################################
        # R2 - Between
        #############################################
        y, x, w = self._prepare_between()
        if np.all(self.weights.values2d == 1.0) and not reweight:
            w = root_w = np.ones_like(w)
        else:
            root_w = cast(Float64Array, np.sqrt(w))
        wx = root_w * x
        wy = root_w * y
        weps = wy - wx @ params
        residual_ss = float(np.squeeze(weps.T @ weps))
        e = y
        if self.has_constant:
            e = y - (w * y).sum() / w.sum()

        total_ss = float(np.squeeze(w.T @ (e**2)))
        r2b = 1 - residual_ss / total_ss if total_ss > 0.0 else 0.0

        #############################################
        # R2 - Overall
        #############################################
        y = self.dependent.values2d
        x = self.exog.values2d
        w = self.weights.values2d
        root_w = cast(Float64Array, np.sqrt(w))
        wx = root_w * x
        wy = root_w * y
        weps = wy - wx @ params
        residual_ss = float(np.squeeze(weps.T @ weps))
        mu = (w * y).sum() / w.sum() if self.has_constant else 0
        we = wy - root_w * mu
        total_ss = float(np.squeeze(we.T @ we))
        r2o = 1 - residual_ss / total_ss if total_ss > 0.0 else 0.0

        #############################################
        # R2 - Within
        #############################################
        weights = self.weights if self._is_weighted else None
        wy = cast(
            Float64Array,
            self.dependent.demean("entity", weights=weights, return_panel=False),
        )
        wx = cast(
            Float64Array,
            self.exog.demean("entity", weights=weights, return_panel=False),
        )
        assert isinstance(wy, np.ndarray)
        assert isinstance(wx, np.ndarray)
        weps = wy - wx @ params
        residual_ss = float(np.squeeze(weps.T @ weps))
        total_ss = float(np.squeeze(wy.T @ wy))
        if self.dependent.nobs == 1 or (self.exog.nvar == 1 and self.has_constant):
            r2w = 0.0
        else:
            r2w = 1.0 - residual_ss / total_ss if total_ss > 0.0 else 0.0

        return r2o, r2w, r2b

    def _postestimation(
        self,
        params: Float64Array,
        cov: CovarianceEstimator,
        debiased: bool,
        df_resid: int,
        weps: Float64Array,
        y: Float64Array,
        x: Float64Array,
        root_w: Float64Array,
    ) -> AttrDict:
        """Common post-estimation values"""
        f_info = self._f_statistic_robust(params)
        f_stat = self._f_statistic(weps, y, x, root_w, df_resid)
        r2o, r2w, r2b = self._rsquared(params)
        c2o, c2w, c2b = self._rsquared_corr(params)
        f_pooled = InapplicableTestStatistic(
            reason="Model has no effects", name="Pooled F-stat"
        )
        entity_info, time_info, other_info = self._info()
        nobs = weps.shape[0]
        sigma2 = float(np.squeeze(weps.T @ weps) / nobs)
        if sigma2 > 0.0:
            loglik = -0.5 * nobs * (np.log(2 * np.pi) + np.log(sigma2) + 1)
        else:
            loglik = np.nan

        res = AttrDict(
            params=params,
            deferred_cov=cov.deferred_cov,
            f_info=f_info,
            f_stat=f_stat,
            debiased=debiased,
            name=self._name,
            var_names=self.exog.vars,
            r2w=r2w,
            r2b=r2b,
            r2=r2w,
            r2o=r2o,
            c2o=c2o,
            c2b=c2b,
            c2w=c2w,
            s2=cov.s2,
            model=self,
            cov_type=cov.name,
            index=self.dependent.index,
            entity_info=entity_info,
            time_info=time_info,
            other_info=other_info,
            f_pooled=f_pooled,
            loglik=loglik,
            not_null=self._not_null,
            original_index=self._original_index,
        )
        return res

    @property
    def not_null(self) -> BoolArray:
        """Locations of non-missing observations"""
        return self._not_null

    def _setup_clusters(
        self,
        cov_config: Mapping[str, bool | float | str | IntArray | DataFrame | PanelData],
    ) -> dict[str, bool | float | str | IntArray | DataFrame | PanelData]:
        cov_config_upd = dict(cov_config)
        cluster_types = ("clusters", "cluster_entity", "cluster_time")
        common = set(cov_config.keys()).intersection(cluster_types)
        if not common:
            return cov_config_upd

        cov_config_upd = {k: v for k, v in cov_config.items()}

        clusters = get_panel_data_like(cov_config, "clusters")
        clusters_frame: DataFrame | None = None
        if clusters is not None:
            formatted_clusters = self.reformat_clusters(clusters)
            for col in formatted_clusters.dataframe:
                cat = Categorical(formatted_clusters.dataframe[col])
                # TODO: Bug in pandas-stubs
                #  https://github.com/pandas-dev/pandas-stubs/issues/111
                formatted_clusters.dataframe[col] = cat.codes.astype(np.int64)  # type: ignore
            clusters_frame = formatted_clusters.dataframe

        cluster_entity = bool(cov_config_upd.pop("cluster_entity", False))
        if cluster_entity:
            group_ids_arr = self.dependent.entity_ids.squeeze()
            name = "cov.cluster.entity"
            group_ids = Series(group_ids_arr, index=self.dependent.index, name=name)
            if clusters_frame is not None:
                clusters_frame[name] = group_ids
            else:
                clusters_frame = DataFrame(group_ids)

        cluster_time = bool(cov_config_upd.pop("cluster_time", False))
        if cluster_time:
            group_ids_arr = self.dependent.time_ids.squeeze()
            name = "cov.cluster.time"
            group_ids = Series(group_ids_arr, index=self.dependent.index, name=name)
            if clusters_frame is not None:
                clusters_frame[name] = group_ids
            else:
                clusters_frame = DataFrame(group_ids)
        if self._singleton_index is not None and clusters_frame is not None:
            clusters_frame = clusters_frame.loc[~self._singleton_index]

        if clusters_frame is not None:
            cov_config_upd["clusters"] = np.asarray(clusters_frame)

        return cov_config_upd

    def predict(
        self,
        params: ArrayLike,
        *,
        exog: PanelDataLike | None = None,
        data: PanelDataLike | None = None,
        eval_env: int = 1,
        context: Mapping[str, Any] | None = None,
    ) -> DataFrame:
        """
        Predict values for additional data

        Parameters
        ----------
        params : array_like
            Model parameters (nvar by 1)
        exog : array_like
            Exogenous regressors (nobs by nvar)
        data : DataFrame
            Values to use when making predictions from a model constructed
            from a formula
        context : int
            Depth of use when evaluating formulas.

        Returns
        -------
        DataFrame
            Fitted values from supplied data and parameters

        Notes
        -----
        If `data` is not None, then `exog` must be None.
        Predictions from models constructed using formulas can
        be computed using either `exog`, which will treat these are
        arrays of values corresponding to the formula-processed data, or using
        `data` which will be processed using the formula used to construct the
        values corresponding to the original model specification.
        """
        if data is not None and self.formula is None:
            raise ValueError(
                "Unable to use data when the model was not " "created using a formula."
            )
        if data is not None and exog is not None:
            raise ValueError(
                "Predictions can only be constructed using one "
                "of exog or data, but not both."
            )
        if exog is not None:
            exog = PanelData(exog).dataframe
        else:
            assert self._formula is not None
            assert data is not None
            if context is None:
                context = capture_context(eval_env)
            parser = PanelFormulaParser(self._formula, data, context=context)
            exog = parser.exog
        x = exog.values
        params = np.atleast_2d(np.asarray(params))
        if params.shape[0] == 1:
            params = params.T
        if x.shape[1] != params.shape[0]:
            raise ValueError(
                EXOG_PREDICT_MSG.format(
                    x_shape=x.shape[1], params_shape=params.shape[0]
                )
            )

        pred = DataFrame(x @ params, index=exog.index, columns=["predictions"])

        return pred

class RandomEffects(_PanelModelBase):
    r"""
    One-way Random Effects model for panel data

    Parameters
    ----------
    dependent : array_like
        Dependent (left-hand-side) variable (time by entity)
    exog : array_like
        Exogenous or right-hand-side variables (variable by time by entity).
    weights : array_like
        Weights to use in estimation.  Assumes residual variance is
        proportional to inverse of weight to that the residual time
        the weight should be homoskedastic.

    Notes
    -----
    The model is given by

    .. math::

        y_{it} = \beta^{\prime}x_{it} + u_i + \epsilon_{it}

    where :math:`u_i` is a shock that is independent of :math:`x_{it}` but
    common to all entities i.
    """

    def __init__(
        self,
        dependent: PanelDataLike,
        exog: PanelDataLike,
        *,
        weights: PanelDataLike | None = None,
        check_rank: bool = True,
    ) -> None:
        super().__init__(dependent, exog, weights=weights, check_rank=check_rank)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: PanelDataLike,
        *,
        weights: PanelDataLike | None = None,
        check_rank: bool = True,
    ) -> RandomEffects:
        """
        Create a model from a formula

        Parameters
        ----------
        formula : str
            Formula to transform into model. Conforms to formulaic formula
            rules.
        data : array_like
            Data structure that can be coerced into a PanelData.  In most
            cases, this should be a multi-index DataFrame where the level 0
            index contains the entities and the level 1 contains the time.
        weights: array_like
            Weights to use in estimation.  Assumes residual variance is
            proportional to inverse of weight to that the residual times
            the weight should be homoskedastic.
        check_rank : bool
            Flag indicating whether to perform a rank check on the exogenous
            variables to ensure that the model is identified. Skipping this
            check can reduce the time required to validate a model
            specification. Results may be numerically unstable if this check
            is skipped and the matrix is not full rank.

        Returns
        -------
        RandomEffects
            Model specified using the formula

        Notes
        -----
        Unlike standard  formula syntax, it is necessary to explicitly include
        a constant using the constant indicator (1)

        Examples
        --------
        >>> from linearmodels import RandomEffects
        >>> from linearmodels.panel import generate_panel_data
        >>> panel_data = generate_panel_data()
        >>> mod = RandomEffects.from_formula("y ~ 1 + x1", panel_data.data)
        >>> res = mod.fit()
        """
        parser = PanelFormulaParser(formula, data, context=capture_context(1))
        dependent, exog = parser.data
        mod = cls(dependent, exog, weights=weights, check_rank=check_rank)
        mod.formula = formula
        return mod

    def fit(
        self,
        *,
        small_sample: bool = False,
        cov_type: str = "unadjusted",
        debiased: bool = True,
        **cov_config: bool | float | str | IntArray | DataFrame | PanelData,
    ) -> RandomEffectsResults:
        """
        Estimate model parameters

        Parameters
        ----------
        small_sample : bool
            Apply a small-sample correction to the estimate of the variance of
            the random effect.
        cov_type : str
            Name of covariance estimator (see notes). Default is "unadjusted".
        debiased : bool
            Flag indicating whether to debiased the covariance estimator using
            a degree of freedom adjustment.
        **cov_config
            Additional covariance-specific options.  See Notes.

        Returns
        -------
        RandomEffectsResults
            Estimation results

        Examples
        --------
        >>> from linearmodels import RandomEffects
        >>> mod = RandomEffects(y, x)
        >>> res = mod.fit(cov_type="clustered", cluster_entity=True)

        Notes
        -----
        Four covariance estimators are supported:

        * "unadjusted", "homoskedastic" - Assume residual are homoskedastic
        * "robust", "heteroskedastic" - Control for heteroskedasticity using
          White's estimator
        * "clustered` - One- or two-way clustering.  Configuration options are:

          * ``clusters`` - Input containing 1 or 2 variables.
            Clusters should be integer values, although other types will
            be coerced to integer values by treating as categorical variables
          * ``cluster_entity`` - Boolean flag indicating to use entity
            clusters
          * ``cluster_time`` - Boolean indicating to use time clusters

        * "kernel" - Driscoll-Kraay HAC estimator. Configurations options are:

          * ``kernel`` - One of the supported kernels (bartlett, parzen, qs).
            Default is Bartlett's kernel, which is produces a covariance
            estimator similar to the Newey-West covariance estimator.
          * ``bandwidth`` - Bandwidth to use when computing the kernel.  If
            not provided, a naive default is used.
        """
        w = self.weights.values2d
        root_w = cast(Float64Array, np.sqrt(w))
        demeaned_dep = self.dependent.demean("entity", weights=self.weights)
        demeaned_exog = self.exog.demean("entity", weights=self.weights)
        assert isinstance(demeaned_dep, PanelData)
        assert isinstance(demeaned_exog, PanelData)
        y = demeaned_dep.values2d
        x = demeaned_exog.values2d
        if self.has_constant:
            w_sum = w.sum()
            y_gm = (w * self.dependent.values2d).sum(0) / w_sum
            x_gm = (w * self.exog.values2d).sum(0) / w_sum
            y += root_w * y_gm
            x += root_w * x_gm
        params = _lstsq(x, y, rcond=None)[0]
        weps = y - x @ params

        wybar = self.dependent.mean("entity", weights=self.weights)
        wxbar = self.exog.mean("entity", weights=self.weights)
        params = _lstsq(np.asarray(wxbar), np.asarray(wybar), rcond=None)[0]
        wu = np.asarray(wybar) - np.asarray(wxbar) @ params

        nobs = weps.shape[0]
        neffects = wu.shape[0]
        nvar = x.shape[1]
        sigma2_e = float(np.squeeze(weps.T @ weps)) / (nobs - nvar - neffects + 1)
        ssr = float(np.squeeze(wu.T @ wu))
        t = np.asarray(self.dependent.count("entity"))
        unbalanced = np.ptp(t) != 0
        if small_sample and unbalanced:
            ssr = float(np.squeeze((t * wu).T @ wu))
            wx_df = cast(DataFrame, root_w * self.exog.dataframe)
            means = wx_df.groupby(level=0).transform("mean").values
            denom = means.T @ means
            sums = wx_df.groupby(level=0).sum().values
            num = sums.T @ sums
            tr = np.trace(np.linalg.inv(denom) @ num)
            sigma2_u = max(0, (ssr - (neffects - nvar) * sigma2_e) / (nobs - tr))
        else:
            t_bar = neffects / ((1.0 / t).sum())
            sigma2_u = max(0, ssr / (neffects - nvar) - sigma2_e / t_bar)
        rho = sigma2_u / (sigma2_u + sigma2_e)

        theta = 1.0 - np.sqrt(sigma2_e / (t * sigma2_u + sigma2_e))
        theta_out = DataFrame(theta, columns=["theta"], index=wybar.index)
        wy: Float64Array = np.asarray(root_w * self.dependent.values2d, dtype=float)
        wx: Float64Array = np.asarray(root_w * self.exog.values2d, dtype=float)
        index = self.dependent.index
        reindex = index.levels[0][index.codes[0]]
        wybar = (theta * wybar).loc[reindex]
        wxbar = (theta * wxbar).loc[reindex]
        wy -= wybar.values
        wx -= wxbar.values
        params = _lstsq(wx, wy, rcond=None)[0]

        df_resid = wy.shape[0] - wx.shape[1]
        cov_config = self._setup_clusters(cov_config)
        extra_df = 0
        if "extra_df" in cov_config:
            cov_config = cov_config.copy()
            _extra_df = cov_config.pop("extra_df")
            assert isinstance(_extra_df, (str, int))
            extra_df = int(_extra_df)
        cov = setup_covariance_estimator(
            self._cov_estimators,
            cov_type,
            wy,
            np.asarray(wx),
            params,
            self.dependent.entity_ids,
            self.dependent.time_ids,
            debiased=debiased,
            extra_df=extra_df,
            **cov_config,
        )

        weps = wy - wx @ params
        eps = weps / root_w
        index = self.dependent.index
        fitted = DataFrame(self.exog.values2d @ params, index, ["fitted_values"])
        effects = DataFrame(
            self.dependent.values2d - np.asarray(fitted) - eps,
            index,
            ["estimated_effects"],
        )
        idiosyncratic = DataFrame(eps, index, ["idiosyncratic"])
        residual_ss = float(np.squeeze(weps.T @ weps))
        wmu: float | Float64Array = 0.0
        if self.has_constant:
            wmu = root_w * _lstsq(root_w, wy, rcond=None)[0]
        wy_demeaned = wy - wmu
        total_ss = float(np.squeeze(wy_demeaned.T @ wy_demeaned))
        r2 = 1 - residual_ss / total_ss

        res = self._postestimation(
            params, cov, debiased, df_resid, weps, wy, wx, root_w
        )
        res.update(
            dict(
                df_resid=df_resid,
                df_model=x.shape[1],
                nobs=y.shape[0],
                residual_ss=residual_ss,
                total_ss=total_ss,
                r2=r2,
                resids=eps,
                wresids=weps,
                index=index,
                sigma2_eps=sigma2_e,
                sigma2_effects=sigma2_u,
                rho=rho,
                theta=theta_out,
                fitted=fitted,
                effects=effects,
                idiosyncratic=idiosyncratic,
            )
        )

        return RandomEffectsResults(res)

