# library/under_development/mixed_effects/centralized_mixed_effect.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

# Single source of truth for node-side summaries
from lmm_reml_node import reml_summaries_node


from common_mixed_effect_functions import (
    build_local_hist,   # (local_max, hist) from nj_sizes
    expand_hist_to_sizes, 
)


# ---------------------------- DF = (X, y, center_ids, w) ---------------------------- #

def build_node_arrays_from_df(
    df: pd.DataFrame,
    *,
    covariate_cols: list[str],
    center_col: str,
    outcome_col: str,
    weights_col: str | None = None,
    include_intercept: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Adapter: convert a pandas DataFrame to the (X, y, center_ids, w) arrays
    expected by `reml_summaries_node`.
    """
    X_df = df[covariate_cols].copy()
    if include_intercept:
        X_df.insert(0, "_intercept", 1.0)
    X = X_df.to_numpy(dtype=float)

    y = df[outcome_col].to_numpy(dtype=float)

    centers = df[center_col].astype("category")
    center_ids = centers.cat.codes.to_numpy(dtype=np.int64)

    w = None
    if weights_col is not None:
        w = df[weights_col].to_numpy(dtype=float)

    return X, y, center_ids, w


# ----------------------------- Centralized to Node payload --------------------------- 

def centralized_lmm_reml_summaries(
    df: pd.DataFrame,
    *,
    covariate_cols: list[str],
    center_col: str,
    outcome_col: str,
    sigma2: float,
    sigma_u2: float,
    beta_hat: np.ndarray,
    weights_col: str | None = None,
    include_intercept: bool = True,
) -> Dict[str, Any]:
    """
    Centralized mirror that produces the SAME payload schema as a single node,
    by delegating all math to `reml_summaries_node`.

    Returns a dict with keys:
      p, Sxx_packed, Sxy, syy, logdet_sum,
      q1_norm_vinv_r_sq, q2_1t_vinv_r_sq,
      T0_tr_vinv_sum, T1_1t_vinv_1,
      B_v_outer_sum_packed, nj_sizes
    """
    X, y, center_ids, w = build_node_arrays_from_df(
        df,
        covariate_cols=covariate_cols,
        center_col=center_col,
        outcome_col=outcome_col,
        weights_col=weights_col,
        include_intercept=include_intercept,
    )

    node_payload = reml_summaries_node(
        X=X,
        y=y,
        center_ids=center_ids,
        sigma2=sigma2,
        sigma_u2=sigma_u2,
        beta_hat=beta_hat,
        w=w,
    )
    return node_payload


# -------------------------- Node payload to Aggregator payload ----------------------- 

def make_aggregator_payload_from_node_payload(
    node_payload: Dict[str, Any],
    *,
    prefer_hist: bool = True,
) -> Dict[str, Any]:
    """
    Convert a node/centralized summaries payload into an aggregator-ready payload.
    - If `prefer_hist` is True (default), compute and attach `global_hist` from `nj_sizes`.
    - Keep `nj_sizes` in the payload too (useful for debug / centralized runs).
      The server-side aggregator will prefer `global_hist` if present.

    This does not alter any packed fields (Sxx_packed, B_v_outer_sum_packed, etc).
    """
    out = dict(node_payload)  

    if prefer_hist:
        nj = node_payload.get("nj_sizes", [])
        # hist from local max
        _, hist = build_local_hist(nj, K=None)
        out["global_hist"] = hist.astype(np.int64)

    return out


# ------------------------------- Convenience end-to-end ----------------------------- 

def centralized_to_aggregator_payload(
    df: pd.DataFrame,
    *,
    covariate_cols: list[str],
    center_col: str,
    outcome_col: str,
    sigma2: float,
    sigma_u2: float,
    beta_hat: np.ndarray,
    weights_col: str | None = None,
    include_intercept: bool = True,
    prefer_hist: bool = True,
) -> Dict[str, Any]:
    """
    One call to go from a DF to an aggregator-ready payload:
      DF → node_payload → (attach global_hist) → agg_payload
    """
    node_payload = centralized_lmm_reml_summaries(
        df,
        covariate_cols=covariate_cols,
        center_col=center_col,
        outcome_col=outcome_col,
        sigma2=sigma2,
        sigma_u2=sigma_u2,
        beta_hat=beta_hat,
        weights_col=weights_col,
        include_intercept=include_intercept,
    )
    return make_aggregator_payload_from_node_payload(node_payload, prefer_hist=prefer_hist)
