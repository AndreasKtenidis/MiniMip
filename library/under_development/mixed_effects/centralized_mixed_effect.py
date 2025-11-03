# library/under_development/mixed_effects/centralized_mixed_effect.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from glmm_laplace_node import glmm_binary_site_derivatives
from lmm_reml_node import reml_summaries_node


from common_mixed_effect_functions import (
    build_local_hist,   
    expand_hist_to_sizes, 
)


# ---------------------------- DF = (X, y, center_ids, w) ---------------------------- 

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


########################## GLMM    ########################


def centralized_glmm_laplace_derivatives(
    df: pd.DataFrame,
    *,
    covariate_cols: list[str],
    center_col: str,
    outcome_col: str,
    theta: np.ndarray,                    
    weights_col: str | None = None,
    include_intercept: bool = True,
    add_laplace_corrections: bool = True,
) -> Dict[str, Any]:
    """
    Centralized node: computes score/H (Laplace) for binary GLMM (random intercept)
    using the entire df as a site.

    Returns a dict compatible with glmm_binary_site_derivatives:
    { "q": q, "score": list[float], "H_packed": list[float],
    "n_centers": int (optional), "sum_logHuu": float (optional) }
    """
    X, y, center_ids, w = build_node_arrays_from_df(
        df,
        covariate_cols=covariate_cols,
        center_col=center_col,
        outcome_col=outcome_col,
        weights_col=weights_col,
        include_intercept=include_intercept,
    )

    theta = np.asarray(theta, dtype=float).reshape(-1)
    msg = glmm_binary_site_derivatives(
        X=X,
        y=y,
        center_ids=center_ids,
        theta=theta,
        w=w,
        add_laplace_corrections=add_laplace_corrections,
    )
    return msg


from glmm_laplace_agg import GLMMAggregator

def run_centralized_glmm_laplace(
    df: pd.DataFrame,
    *,
    covariate_cols: list[str],
    center_col: str,
    outcome_col: str,
    theta0: np.ndarray | None = None,     
    weights_col: str | None = None,
    include_intercept: bool = True,
    add_laplace_corrections: bool = True,
    max_iters: int = 50,
    ridge: float = 1e-6,
    tol_theta: float = 1e-6,
    tol_score: float = 1e-4,
    clip_log_sigma_bounds: tuple[float, float] = (np.log(1e-8), np.log(1e+3)),
    max_step_norm: float = 5.0,
    beta_init_zero: bool = True,          
    log_sigma_u2_init: float = np.log(0.3),
) -> Dict[str, Any]:
    """
    Centralized Newton-Laplace for binary GLMM (random intercept).

    Returns:
    {
    "theta": θ_final, "beta": β_final, "sigma_u2": σ_u^2_final,
    "iters": k, "converged": bool, "history": [ ... ],
    "last_derivatives": { "score": ..., "H_packed": ... }
    }
    """
    X, y, center_ids, w = build_node_arrays_from_df(
        df,
        covariate_cols=covariate_cols,
        center_col=center_col,
        outcome_col=outcome_col,
        weights_col=weights_col,
        include_intercept=include_intercept,
    )
    n, p = X.shape

    # init theta
    if theta0 is None:
        if beta_init_zero:
            beta0 = np.zeros(p, dtype=float)
        else:
            beta0 = (1.0 / max(np.sqrt(n), 1.0)) * np.ones(p, dtype=float)
        theta = np.concatenate([beta0, np.array([log_sigma_u2_init], dtype=float)])
    else:
        theta = np.asarray(theta0, dtype=float).reshape(-1)
        if theta.shape[0] != p + 1:
            raise ValueError(f"theta0 must have length p+1={p+1}, got {theta.shape[0]}")

    agg = GLMMAggregator(q=p + 1)
    hist: list[Dict[str, Any]] = []
    converged = False
    last_derivatives: Dict[str, Any] = {}

    for it in range(1, max_iters + 1):
        # 1) node derivatives 
        msg = centralized_glmm_laplace_derivatives(
            df,
            covariate_cols=covariate_cols,
            center_col=center_col,
            outcome_col=outcome_col,
            theta=theta,
            weights_col=weights_col,
            include_intercept=include_intercept,
            add_laplace_corrections=add_laplace_corrections,
        )

        # 2) aggregator global sums 
        agg.reset()
        agg.accumulate({
            "score": msg["score"],
            "H_packed": msg["H_packed"],
        })

        theta_new = agg.newton_update(
            theta,
            ridge=ridge,
            max_tries=6,
            max_step_norm=max_step_norm,
            clip_log_sigma_bounds=clip_log_sigma_bounds,
        )

        score_norm = float(np.linalg.norm(agg.s))
        dtheta_max = float(np.max(np.abs(theta_new - theta)))
        hist.append({
            "iter": it,
            "score_norm": score_norm,
            "dtheta_max": dtheta_max,
        })

        if GLMMAggregator.converged(theta, theta_new, agg.s, tol_theta=tol_theta, tol_score=tol_score):
            theta = theta_new
            converged = True
            last_derivatives = {
                "score": np.asarray(msg["score"], dtype=float),
                "H_packed": np.asarray(msg["H_packed"], dtype=float),
                "n_centers": msg.get("n_centers", None),
                "sum_logHuu": msg.get("sum_logHuu", None),
            }
            break

        theta = theta_new
        last_derivatives = {
            "score": np.asarray(msg["score"], dtype=float),
            "H_packed": np.asarray(msg["H_packed"], dtype=float),
            "n_centers": msg.get("n_centers", None),
            "sum_logHuu": msg.get("sum_logHuu", None),
        }

    beta = theta[:p]
    sigma_u2 = float(np.exp(theta[p]))

    return {
        "theta": theta,
        "beta": beta,
        "sigma_u2": sigma_u2,
        "iters": len(hist),
        "converged": bool(converged),
        "history": hist,
        "last_derivatives": last_derivatives,
    }
