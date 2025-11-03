# mini_mip_system/client/glmm_laplace_runner.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
import numpy as np

from mini_mip_system.client.grpc_agg_client import GRPCClient
from utils.numpy_aggregator import NumpyAggregator  # ή library.utils.numpy_aggregator ανάλογα με το project σου

from glmm_laplace_node import glmm_binary_site_derivatives
from glmm_laplace_agg import GLMMAggregator


def run_glmm_laplace(
    client: GRPCClient,
    *,
    X: np.ndarray,
    y: np.ndarray,
    center_ids: np.ndarray,
    theta0: np.ndarray,                     
    w: Optional[np.ndarray] = None,
    max_iters: int = 50,
    ridge: float = 1e-6,
    tol_theta: float = 1e-6,
    tol_score: float = 1e-4,
    add_laplace_corrections: bool = True,
    clip_log_sigma_bounds: Optional[tuple[float, float]] = (np.log(1e-8), np.log(1e+3)),
    max_step_norm: float = 5.0,
) -> Dict[str, Any]:
    """
    Federated GLMM (binary, random intercept) with Laplace approximation.
    At each iteration:
    - Node: glmm_binary_site_derivatives(θ_k) → (score, H_packed)
    - Coordinator: fed-sum score/H, Newton update (damped), θ_{k+1}
    - Consensus: broadcast θ_{k+1} via fed_sum (coordinator sends, others 0)
    Stops when: max|Δθ| < tol_theta AND ||score||_2 < tol_score, or max_iters reached.

    Returns:
    {
    "theta": θ_final,
    "beta": β_final,
    "sigma_u2": σ_u^2_final,
    "iters": k,
    "converged": bool,
    "history": ,
    "aggregates":  # last iteration
    }
    """
    npagg = NumpyAggregator(client)

    theta = np.asarray(theta0, dtype=np.float64).reshape(-1)
    q = theta.shape[0]
    p = X.shape[1]
    if q != p + 1:
        raise ValueError(f"theta0 length ({q}) must be p+1 ({p+1}) for [beta..., log_sigma_u2].")

    hist: List[Dict[str, Any]] = []
    converged = False
    last_aggregates: Dict[str, Any] = {}

    for it in range(1, max_iters + 1):
        # ------------------------ Node derivatives  ------------------------ 
        site_msg = glmm_binary_site_derivatives(
            X=X,
            y=y,
            center_ids=center_ids,
            theta=theta,
            w=w,
            add_laplace_corrections=add_laplace_corrections,
        )
        score_local = np.asarray(site_msg["score"], dtype=np.float64)
        H_packed_local = np.asarray(site_msg["H_packed"], dtype=np.float64)

        
        n_centers_local = np.array([site_msg.get("n_centers", 0.0)], dtype=np.float64)
        sum_logHuu_local = np.array([site_msg.get("sum_logHuu", 0.0)], dtype=np.float64)

        # ------------------------ Federated SUMs (score, H_packed) -------------------- 
        score_sum = npagg.fed_sum(score_local)
        H_packed_sum = npagg.fed_sum(H_packed_local)

        
        n_centers_global = int(npagg.fed_sum(n_centers_local)[0])
        sum_logHuu_global = float(npagg.fed_sum(sum_logHuu_local)[0])

        # ------------------------ Aggregation & Newton step --------------------------- 
        agg = GLMMAggregator(q)
        agg.reset()
       
        agg.accumulate({
            "score": score_sum,
            "H_packed": H_packed_sum,
        })

        theta_prop = agg.newton_update(
            theta,
            ridge=ridge,
            max_tries=6,
            max_step_norm=max_step_norm,
            clip_log_sigma_bounds=clip_log_sigma_bounds,
        )

        # ------------------------ Consensus broadcast of θ_{k+1} ---------------------- #
        is_coord = getattr(client, "client_id", 0) == 0
        theta_send = theta_prop if is_coord else np.zeros_like(theta_prop)
        theta_next = npagg.fed_sum(theta_send)   # all nodes  receive the same θ

        # ------------------------ Convergence check ---------------------------------- #
        score_norm = float(np.linalg.norm(agg.s))
        dtheta_max = float(np.max(np.abs(theta_next - theta)))
        hist.append({
            "iter": it,
            "score_norm": score_norm,
            "dtheta_max": dtheta_max,
            "theta": theta_next.copy(),
            "n_centers": n_centers_global,
            "sum_logHuu": sum_logHuu_global,
        })

        if GLMMAggregator.converged(theta, theta_next, agg.s, tol_theta=tol_theta, tol_score=tol_score):
            theta = theta_next
            converged = True
            last_aggregates = {
                "score_sum": score_sum,
                "H_packed_sum": H_packed_sum,
                "n_centers": n_centers_global,
                "sum_logHuu": sum_logHuu_global,
            }
            break

        # prepare next iteration
        theta = theta_next
        last_aggregates = {
            "score_sum": score_sum,
            "H_packed_sum": H_packed_sum,
            "n_centers": n_centers_global,
            "sum_logHuu": sum_logHuu_global,
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
        "aggregates": last_aggregates,
    }
