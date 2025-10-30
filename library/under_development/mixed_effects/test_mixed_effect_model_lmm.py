# tests/test_lmm_reml_fed_vs_cent.py
from __future__ import annotations
import numpy as np
import pandas as pd

from lmm_reml_aggregator import LMMRemlAggregator
from lmm_reml_node import reml_summaries_node
from centralized_mixed_effect import (
    build_node_arrays_from_df,
    centralized_lmm_reml_summaries,
)
from common_mixed_effect_functions import (
    build_local_hist, expand_hist_to_sizes,
    pack_upper_triangle, unpack_upper_triangle,
)

# 1) Synth data generator 

def synth_lmm_df(
    n_centers=8,
    n_features=2,
    n_min=20, n_max=60,
    beta=None,                   
    sigma2=1.0,
    sigma_u2=0.5,
    seed=123,
):
    rng = np.random.default_rng(seed)
    if beta is None:
        beta = np.array([0.3] + [0.5]*n_features, dtype=float)

    rows = []
    centers_order = []
    for j in range(n_centers):
        n_j = rng.integers(n_min, n_max+1)
        X_cov = rng.normal(size=(n_j, n_features))
        intercept = np.ones((n_j, 1))
        X = np.hstack([intercept, X_cov])
        u_j = rng.normal(loc=0.0, scale=np.sqrt(sigma_u2))
        eps = rng.normal(loc=0.0, scale=np.sqrt(sigma2), size=n_j)
        y = X @ beta + u_j + eps
        c = np.full(n_j, f"C{j}")
        centers_order.append((f"C{j}", int(n_j)))
        rows.append(pd.DataFrame({"center": c, "y": y, **{f"x{k+1}": X_cov[:, k] for k in range(n_features)}}))
    df = pd.concat(rows, ignore_index=True)
    return df, beta, sigma2, sigma_u2, centers_order

# 2) Centralized 

def run_centralized(df, covs, center_col="center", outcome_col="y",
                    sigma2=1.0, sigma_u2=0.5, beta_hat=None):
    X, _, _, _ = build_node_arrays_from_df(
        df, covariate_cols=covs, center_col=center_col, outcome_col=outcome_col, include_intercept=True
    )
    p = X.shape[1]
    if beta_hat is None:
        beta_hat = np.zeros(p, dtype=float)

    node_payload = centralized_lmm_reml_summaries(
        df,
        covariate_cols=covs,
        center_col=center_col,
        outcome_col=outcome_col,
        sigma2=sigma2,
        sigma_u2=sigma_u2,
        beta_hat=beta_hat,
        include_intercept=True,
    )

    agg = LMMRemlAggregator(p)
    agg.reset()
    agg.accumulate({
        "Sxx_packed": node_payload["Sxx_packed"],
        "Sxy":        node_payload["Sxy"],
        "syy":        node_payload["syy"],
        "q1_norm_vinv_r_sq": node_payload["q1_norm_vinv_r_sq"],
        "q2_1t_vinv_r_sq":   node_payload["q2_1t_vinv_r_sq"],
        "T0_tr_vinv_sum":    node_payload["T0_tr_vinv_sum"],
        "T1_1t_vinv_1":      node_payload["T1_1t_vinv_1"],
        "B_v_outer_sum_packed": node_payload["B_v_outer_sum_packed"],
        "nj_sizes": node_payload["nj_sizes"],   
    })
    beta_gls, Ainv = agg.compute_beta_gls()
    s2_new, su2_new, improved = agg.update_variances(beta_gls, Ainv, sigma2, sigma_u2)
    return {
        "p": p,
        "payload": node_payload,
        "beta": beta_gls,
        "sigma2": s2_new,
        "sigma_u2": su2_new,
        "improved": improved,
        "agg": agg,
    }

# 3) Federated emulation 

def split_df_into_nodes(df, k_nodes=3, seed=123):
    rng = np.random.default_rng(seed)
    centers = df["center"].unique().tolist()
    rng.shuffle(centers)
    bins = [centers[i::k_nodes] for i in range(k_nodes)]
    node_dfs = [df[df["center"].isin(b)].copy() for b in bins]
    return node_dfs, bins

def run_federated_emulated(df, covs, k_nodes=3, center_col="center", outcome_col="y",
                           sigma2=1.0, sigma_u2=0.5, beta_hat=None):
    X, _, _, _ = build_node_arrays_from_df(
        df, covariate_cols=covs, center_col=center_col, outcome_col=outcome_col, include_intercept=True
    )
    p = X.shape[1]
    if beta_hat is None:
        beta_hat = np.zeros(p, dtype=float)

    node_dfs, bins = split_df_into_nodes(df, k_nodes=k_nodes)

    node_payloads = []
    local_max_list = []
    local_hists = []
    node_nj_sizes = []

    for ndf in node_dfs:
        Xn, yn, cn, wn = build_node_arrays_from_df(
            ndf, covariate_cols=covs, center_col=center_col, outcome_col=outcome_col, include_intercept=True
        )
        pay = reml_summaries_node(
            X=Xn, y=yn, center_ids=cn,
            sigma2=sigma2, sigma_u2=sigma_u2,
            beta_hat=beta_hat, w=wn
        )
        node_payloads.append(pay)
        node_nj_sizes.append(pay["nj_sizes"])
        loc_max, _ = build_local_hist(pay["nj_sizes"])
        local_max_list.append(loc_max)

    K = int(max(local_max_list)) if local_max_list else 0

    if K > 0:
        for pay in node_payloads:
            _, loc_hist = build_local_hist(pay["nj_sizes"], K=K)
            local_hists.append(loc_hist)
        global_hist = np.sum(np.vstack(local_hists), axis=0).astype(np.int64)
    else:
        global_hist = np.zeros(0, dtype=np.int64)

    def sum_lists(name):
        arrs = [np.asarray(pay[name], dtype=np.float64) for pay in node_payloads]
        return np.sum(np.vstack(arrs), axis=0)

    Sxx_sum = sum_lists("Sxx_packed")
    Sxy_sum = sum_lists("Sxy")
    syy_sum = float(np.sum([pay["syy"] for pay in node_payloads]))
    q1 = float(np.sum([pay["q1_norm_vinv_r_sq"] for pay in node_payloads]))
    q2 = float(np.sum([pay["q2_1t_vinv_r_sq"] for pay in node_payloads]))
    T0 = float(np.sum([pay["T0_tr_vinv_sum"] for pay in node_payloads]))
    T1 = float(np.sum([pay["T1_1t_vinv_1"] for pay in node_payloads]))
    B_sum  = sum_lists("B_v_outer_sum_packed")

    agg = LMMRemlAggregator(p)
    agg.reset()
    agg.accumulate({
        "Sxx_packed": Sxx_sum.tolist(),
        "Sxy":        Sxy_sum.tolist(),
        "syy":        syy_sum,
        "q1_norm_vinv_r_sq": q1,
        "q2_1t_vinv_r_sq":   q2,
        "T0_tr_vinv_sum":    T0,
        "T1_1t_vinv_1":      T1,
        "B_v_outer_sum_packed": B_sum.tolist(),
        "global_hist": global_hist,
    })
    beta_gls, Ainv = agg.compute_beta_gls()
    s2_new, su2_new, improved = agg.update_variances(beta_gls, Ainv, sigma2, sigma_u2)

    return {
        "p": p,
        "beta": beta_gls,
        "sigma2": s2_new,
        "sigma_u2": su2_new,
        "improved": improved,
        "global_hist": global_hist,
        "Sxx_packed_sum": Sxx_sum,
        "Sxy_sum": Sxy_sum,
        "syy_sum": syy_sum,
        "q_vec_sum": np.array([q1, q2, T0, T1], dtype=np.float64),
        "B_packed_sum": B_sum,
        "node_splits": bins,
        "node_nj_sizes": node_nj_sizes,
    }

#  4) Compare 

def assert_allclose(a, b, rtol=1e-8, atol=1e-10, msg=""):
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        diff = np.max(np.abs(a - b))
        raise AssertionError(f"{msg} max|Δ|={diff}  (rtol={rtol}, atol={atol})")

def print_header(title: str):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def arr_str(a):
    return np.array2string(np.asarray(a), precision=6, floatmode="fixed", suppress_small=False)

# ---------- 5) Run test with full reporting ----------

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=160)

   
    n_centers = 8
    n_features = 2
    n_min, n_max = 30, 60
    s2_true, su2_true = 1.2, 0.7
    beta0 = None  
    s2_0, su2_0 = 1.0, 0.5
    k_nodes = 3

    
    df, beta_true, s2_true, su2_true, centers_info = synth_lmm_df(
        n_centers=n_centers, n_features=n_features,
        n_min=n_min, n_max=n_max,
        sigma2=s2_true, sigma_u2=su2_true, seed=42
    )
    covs = ["x1", "x2"]

    print_header("INPUT / DATA SUMMARY")
    print(f"n_centers = {n_centers}, n_features = {n_features}, patients per center in [{n_min}, {n_max}]")
    print(f"True beta (used to generate y): {arr_str(beta_true)}")
    print(f"True variances: sigma2={s2_true:.6f}, sigma_u2={su2_true:.6f}")
    print("Centers & sizes:", centers_info)

    # Centralized
    C = run_centralized(df, covs, sigma2=s2_0, sigma_u2=su2_0, beta_hat=beta0)
    # Federated emulation
    F = run_federated_emulated(df, covs, k_nodes=k_nodes, sigma2=s2_0, sigma_u2=su2_0, beta_hat=beta0)

    p = C["p"]

    
    Sxx_c = unpack_upper_triangle(C["payload"]["Sxx_packed"], p)
    Sxx_f = unpack_upper_triangle(F["Sxx_packed_sum"].tolist(), p)
    B_c   = unpack_upper_triangle(C["payload"]["B_v_outer_sum_packed"], p)
    B_f   = unpack_upper_triangle(F["B_packed_sum"].tolist(), p)

    print_header("CENTRALIZED SUMMARY (node-style payload)")
    print(f"p = {p}")
    print(f"Sxx (centralized):\n{arr_str(Sxx_c)}")
    print(f"Sxy (centralized): {arr_str(C['payload']['Sxy'])}")
    print(f"syy (centralized): {C['payload']['syy']:.10f}")
    print(f"q vector (centralized): {arr_str([C['payload']['q1_norm_vinv_r_sq'], C['payload']['q2_1t_vinv_r_sq'], C['payload']['T0_tr_vinv_sum'], C['payload']['T1_1t_vinv_1']])}")
    print(f"B (centralized):\n{arr_str(B_c)}")
    print(f"nj_sizes (centralized): {C['payload']['nj_sizes']}")

    print_header("FEDERATED SUMMARY (after SUM/MAX)")
    print(f"global_hist (federated): {arr_str(F['global_hist'])}  (sum bins = total centers)")
    print(f"Sxx_sum (federated):\n{arr_str(Sxx_f)}")
    print(f"Sxy_sum (federated): {arr_str(F['Sxy_sum'])}")
    print(f"syy_sum (federated): {F['syy_sum']:.10f}")
    print(f"q_vec_sum (federated): {arr_str(F['q_vec_sum'])}")
    print(f"B_packed_sum (→unpacked):\n{arr_str(B_f)}")
    print(f"node_splits (centers per node): {F['node_splits']}")
    print(f"per-node nj_sizes: {F['node_nj_sizes']}")

    # Diffs on raw summaries
    print_header("DIFFERENCES ON SUMMARIES (centralized - federated)")
    print("ΔSxx:\n", arr_str(Sxx_c - Sxx_f))
    print("ΔSxy: ", arr_str(np.array(C["payload"]["Sxy"]) - np.array(F["Sxy_sum"])))
    print("Δsyy: ", f"{C['payload']['syy'] - F['syy_sum']:.12f}")
    q_c = np.array([
        C["payload"]["q1_norm_vinv_r_sq"],
        C["payload"]["q2_1t_vinv_r_sq"],
        C["payload"]["T0_tr_vinv_sum"],
        C["payload"]["T1_1t_vinv_1"],
    ], dtype=np.float64)
    print("Δq:   ", arr_str(q_c - F["q_vec_sum"]))
    print("ΔB:\n", arr_str(B_c - B_f))

    # GLS & REML updates
    print_header("GLS / REML RESULTS")
    print("Centralized:")
    print("  beta_gls:", arr_str(C["beta"]))
    print(f"  sigma2_new={C['sigma2']:.10f}, sigma_u2_new={C['sigma_u2']:.10f}, improved={C['improved']}")

    print("Federated:")
    print("  beta_gls:", arr_str(F["beta"]))
    print(f"  sigma2_new={F['sigma2']:.10f}, sigma_u2_new={F['sigma_u2']:.10f}, improved={F['improved']}")

    print_header("DIFFERENCES ON RESULTS (centralized - federated)")
    print("Δbeta_gls:", arr_str(C["beta"] - F["beta"]))
    print(f"Δsigma2: {C['sigma2'] - F['sigma2']:.12f}")
    print(f"Δsigma_u2: {C['sigma_u2'] - F['sigma_u2']:.12f}")

   
    def assert_allclose(a, b, rtol=1e-8, atol=1e-10, msg=""):
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            diff = np.max(np.abs(a - b))
            raise AssertionError(f"{msg} max|Δ|={diff}  (rtol={rtol}, atol={atol})")

    
    assert_allclose(Sxx_c, Sxx_f, msg="Sxx mismatch")
    assert_allclose(np.array(C["payload"]["Sxy"]), np.array(F["Sxy_sum"]), msg="Sxy mismatch")
    assert_allclose(C["payload"]["syy"], F["syy_sum"], msg="syy mismatch")
    assert_allclose(q_c, F["q_vec_sum"], msg="q vector mismatch")
    assert_allclose(B_c, B_f, msg="B matrix mismatch")
    assert_allclose(C["beta"], F["beta"], rtol=1e-10, msg="beta_gls mismatch")
    assert_allclose(C["sigma2"], F["sigma2"], rtol=1e-7, msg="sigma2 update mismatch")
    assert_allclose(C["sigma_u2"], F["sigma_u2"], rtol=1e-7, msg="sigma_u2 update mismatch")

    print_header("  Centralized vs Federated emulation match within tolerances.")
