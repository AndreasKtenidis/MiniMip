# tests/test_glmm_laplace_fed_vs_cent.py
from __future__ import annotations
import numpy as np
import pandas as pd

from centralized_mixed_effect import (
    build_node_arrays_from_df,
    run_centralized_glmm_laplace,
)
from glmm_laplace_node import glmm_binary_site_derivatives
from glmm_laplace_agg import GLMMAggregator
from common_mixed_effect_functions import unpack_upper_triangle, pack_upper_triangle


from library.utils.numpy_aggregator import NumpyAggregator  


# --------------------------- utilities / pretty print ---------------------------

def print_header(title: str):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def arr_str(a):
    return np.array2string(np.asarray(a), precision=6, floatmode="fixed", suppress_small=False)

def assert_allclose(a, b, rtol=1e-8, atol=1e-10, msg=""):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        diff = float(np.max(np.abs(a - b)))
        raise AssertionError(f"{msg} max|Δ|={diff} (rtol={rtol}, atol={atol})")


# --------------------------- synthetic data generator ---------------------------

def synth_glmm_binary_df(
    n_centers=15,
    n_features=2,
    n_min=40, n_max=80,
    beta=None,                  
    sigma_u2=0.6,
    seed=42,
):
    rng = np.random.default_rng(seed)
    if beta is None:
        beta = np.array([-0.3] + [0.6]*n_features, dtype=float)

    rows = []
    centers_info = []
    for j in range(n_centers):
        n_j = int(rng.integers(n_min, n_max+1))
        X_cov = rng.normal(size=(n_j, n_features))
        intercept = np.ones((n_j, 1))
        X = np.hstack([intercept, X_cov])

        u_j = rng.normal(loc=0.0, scale=np.sqrt(sigma_u2))
        eta = X @ beta + u_j
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(n=1, p=p, size=n_j)

        c = np.full(n_j, f"C{j}")
        centers_info.append((f"C{j}", n_j))
        rows.append(pd.DataFrame({"center": c, "y": y, **{f"x{k+1}": X_cov[:, k] for k in range(n_features)}}))

    df = pd.concat(rows, ignore_index=True)
    return df, beta, sigma_u2, centers_info


# --------------------------- federated split ---------------------------

def split_df_into_nodes(df: pd.DataFrame, k_nodes=3, seed=123):
    rng = np.random.default_rng(seed)
    centers = df["center"].unique().tolist()
    rng.shuffle(centers)
    bins = [centers[i::k_nodes] for i in range(k_nodes)]
    node_dfs = [df[df["center"].isin(b)].copy() for b in bins]
    return node_dfs, bins


# --------------------------- federated emulation loop ---------------------------

def run_federated_emulated_glmm_laplace(
    df: pd.DataFrame,
    covs: list[str],
    *,
    k_nodes=3,
    center_col="center",
    outcome_col="y",
    theta0: np.ndarray | None = None,     
    max_iters: int = 50,
    ridge: float = 1e-6,
    tol_theta: float = 1e-6,
    tol_score: float = 1e-4,
    add_laplace_corrections: bool = True,
    clip_log_sigma_bounds=(np.log(1e-8), np.log(1e+3)),
    max_step_norm: float = 5.0,
):
    """
    Federated emulation:
    - splits DF into nodes (per center),
    - computes site-derivatives per node,
    - does federated SUM (via NumpyAggregator for API consistency),
    - Newton step with GLMMAggregator,
    - iterates until convergence.
    """
    
    class _DummyClient:
        client_id = 0
    npagg = NumpyAggregator(_DummyClient())

   
    node_dfs, bins = split_df_into_nodes(df, k_nodes=k_nodes)

   
    X0, _, _, _ = build_node_arrays_from_df(node_dfs[0], covariate_cols=covs, center_col=center_col, outcome_col=outcome_col, include_intercept=True)
    p = X0.shape[1]

   
    if theta0 is None:
        beta0 = np.zeros(p, dtype=float)
        theta = np.concatenate([beta0, np.array([np.log(0.3)], dtype=float)])
    else:
        theta = np.asarray(theta0, dtype=float).reshape(-1)
        if theta.shape[0] != p + 1:
            raise ValueError(f"theta0 must have length p+1={p+1}, got {theta.shape[0]}")

    agg = GLMMAggregator(q=p+1)
    hist = []

    for it in range(1, max_iters+1):
       
        local_scores = []
        local_Hs = []
        for ndf in node_dfs:
            Xn, yn, cn, wn = build_node_arrays_from_df(
                ndf, covariate_cols=covs, center_col=center_col, outcome_col=outcome_col, include_intercept=True
            )
            payload = glmm_binary_site_derivatives(
                X=Xn, y=yn, center_ids=cn, theta=theta, w=wn,
                add_laplace_corrections=add_laplace_corrections
            )
            local_scores.append(np.asarray(payload["score"], dtype=float))
            Hn = unpack_upper_triangle(np.asarray(payload["H_packed"], dtype=float), p+1)
            local_Hs.append(Hn)

        # federated SUMs 
        score_sum_local = np.sum(np.vstack(local_scores), axis=0)
        H_sum_local = np.sum(np.stack(local_Hs, axis=0), axis=0)

        score_sum = npagg.fed_sum(score_sum_local)        
        H_packed_sum = npagg.fed_sum(pack_upper_triangle(H_sum_local))

        # Aggregator step 
        agg.reset()
        agg.accumulate({
            "score": score_sum,
            "H_packed": H_packed_sum,
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
        hist.append({"iter": it, "score_norm": score_norm, "dtheta_max": dtheta_max})

        if GLMMAggregator.converged(theta, theta_new, agg.s, tol_theta=tol_theta, tol_score=tol_score):
            theta = theta_new
            break

        theta = theta_new

    beta = theta[:p]
    sigma_u2 = float(np.exp(theta[p]))
    return {
        "theta": theta, "beta": beta, "sigma_u2": sigma_u2,
        "iters": len(hist), "history": hist,
        "node_splits": bins,
    }


# ------------------------------------ MAIN ------------------------------------

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=160)

    
    n_centers, n_features = 15, 2
    n_min, n_max = 40, 80
    sigma_u2_true = 0.6

    df, beta_true, su2_true, centers_info = synth_glmm_binary_df(
        n_centers=n_centers, n_features=n_features,
        n_min=n_min, n_max=n_max,
        sigma_u2=sigma_u2_true, seed=42
    )
    covs = ["x1", "x2"]

    print_header("INPUT / DATA SUMMARY")
    print(f"n_centers={n_centers}, n_features={n_features}, obs per center in [{n_min}, {n_max}]")
    print(f"True beta: {arr_str(beta_true)}")
    print(f"True sigma_u2: {su2_true:.6f}")
    print("Centers & sizes:", centers_info)

    # centralized 
    C = run_centralized_glmm_laplace(
        df,
        covariate_cols=covs,
        center_col="center",
        outcome_col="y",
        theta0=None,
        max_iters=50,
        ridge=1e-6,
        tol_theta=1e-6,
        tol_score=1e-4,
        add_laplace_corrections=True,
        clip_log_sigma_bounds=(np.log(1e-8), np.log(1e+3)),
        max_step_norm=5.0,
    )

    # federated 
    F = run_federated_emulated_glmm_laplace(
        df, covs,
        k_nodes=3,
        center_col="center",
        outcome_col="y",
        theta0=None,
        max_iters=50,
        ridge=1e-6,
        tol_theta=1e-6,
        tol_score=1e-4,
        add_laplace_corrections=True,
        clip_log_sigma_bounds=(np.log(1e-8), np.log(1e+3)),
        max_step_norm=5.0,
    )

    p = len(beta_true)

    print_header("RESULTS")
    print("Centralized:")
    print("  beta:", arr_str(C["beta"]))
    print(f"  sigma_u2={C['sigma_u2']:.6f}, iters={C['iters']}")
    print("Federated:")
    print("  beta:", arr_str(F["beta"]))
    print(f"  sigma_u2={F['sigma_u2']:.6f}, iters={F['iters']}")

    print_header("DIFFERENCES (centralized - federated)")
    print("Δbeta:", arr_str(C["beta"] - F["beta"]))
    print(f"Δsigma_u2: {C['sigma_u2'] - F['sigma_u2']:.12f}")

    # centralized vs federated 
    assert_allclose(C["beta"], F["beta"], rtol=1e-8, atol=1e-8, msg="beta (C vs F)")
    assert_allclose(C["sigma_u2"], F["sigma_u2"], rtol=1e-7, atol=1e-8, msg="sigma_u2 (C vs F)")

    print_header("  Centralized vs Federated emulation match within tolerances.")
