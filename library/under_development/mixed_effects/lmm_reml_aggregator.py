# mini_mip_system/models/lmm_reml_aggregator.py
from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np

from common_mixed_effect_functions import unpack_upper_triangle  
from common_mixed_effect_functions import (
    gls_beta_from_Sxx_Sxy,
    reml_objective_from_summaries,
    reml_grad_logscale_from_summaries,
    backtracking_line_search_log2d,
)

class LMMRemlAggregator:
    """
    Stateful accumulator + optimizer for LMM REML over federated summaries.

    Holds global (summed) quantities:
      Sxx (p x p), Sxy (p,), syy (scalar),
      q1, q2, T0, T1 (scalars), B (p x p),
      and either global_hist (np.ndarray of counts) OR nj_sizes (list[int]).
      If both are present, global_hist takes precedence.
    """

    def __init__(self, p: int):
        self.p = int(p)
        self.reset()

    # ------------------------- state / accumulation ------------------------- #

    def reset(self) -> None:
        """Reset all global accumulators."""
        self.Sxx = np.zeros((self.p, self.p), dtype=np.float64)
        self.Sxy = np.zeros(self.p, dtype=np.float64)
        self.syy = 0.0
        self.logdet = 0.0  # if needed in future, replacing hist

        # REML helpers
        self.q1 = 0.0
        self.q2 = 0.0
        self.T0 = 0.0
        self.T1 = 0.0
        self.B  = np.zeros((self.p, self.p), dtype=np.float64)

        # cluster-size info (use either; hist has precedence)
        self.global_hist: np.ndarray | None = None
        self.nj_sizes: list[int] | None = []

    def accumulate(self, payload: Dict[str, Any]) -> None:
        """
        Accumulate one (already federated-summed) payload into the global state.

        Required keys (if present they get added):
          - "Sxx_packed": list[float] upper-triangular
          - "Sxy": list[float] length p
          - "syy": float
          - "q1_norm_vinv_r_sq", "q2_1t_vinv_r_sq", "T0_tr_vinv_sum", "T1_1t_vinv_1": floats
          - "B_v_outer_sum_packed": list[float] upper-triangular
          - cluster size info (one of):
              * "global_hist": list[int] or np.ndarray
              * "nj_sizes": list[int]  (used only if global_hist not provided)
        """
        if not isinstance(payload, dict):
            raise TypeError("accumulate expects a dict payload")

        # Sxx, Sxy, syy
        sxx_packed = payload.get("Sxx_packed")
        if sxx_packed is not None:
            self.Sxx += unpack_upper_triangle(sxx_packed, self.p)

        sxy = payload.get("Sxy")
        if sxy is not None:
            self.Sxy += np.asarray(sxy, dtype=np.float64)

        syy = payload.get("syy")
        if syy is not None:
            self.syy += float(syy)

        #  diagnostic
        logdet_sum = payload.get("logdet_sum")
        if logdet_sum is not None:
            self.logdet += float(logdet_sum)

        # REML score ingredients
        q1 = payload.get("q1_norm_vinv_r_sq")
        if q1 is not None:
            self.q1 += float(q1)

        q2 = payload.get("q2_1t_vinv_r_sq")
        if q2 is not None:
            self.q2 += float(q2)

        T0 = payload.get("T0_tr_vinv_sum")
        if T0 is not None:
            self.T0 += float(T0)

        T1 = payload.get("T1_1t_vinv_1")
        if T1 is not None:
            self.T1 += float(T1)

        B_packed = payload.get("B_v_outer_sum_packed")
        if B_packed is not None and len(B_packed) > 0:
            self.B += unpack_upper_triangle(B_packed, self.p)

        
        hist = payload.get("global_hist")
        if hist is not None:
            self.global_hist = np.asarray(hist, dtype=np.int64)
            self.nj_sizes = None  # when nj_sizes=none then we calculate histogram mode
        else:
            sizes = payload.get("nj_sizes")
            if sizes is not None:
                if self.nj_sizes is None:
                    self.nj_sizes = list(sizes)
                else:
                    self.nj_sizes.extend(list(sizes))

    # ------------------------- GLS / objective / gradient ------------------------- #

    def compute_beta_gls(self, ridge: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GLS beta and Ainv from accumulated Sxx, Sxy."""
        return gls_beta_from_Sxx_Sxy(self.Sxx, self.Sxy, ridge=ridge)

    def objective(self, beta: np.ndarray, sigma2: float, sigma_u2: float) -> float:
        """
        ℓ_REML from global summaries.
        Uses global_hist if set; otherwise falls back to nj_sizes.
        """
        if self.global_hist is not None:
            return reml_objective_from_summaries(
                self.Sxx, self.syy, beta,
                sigma2=sigma2, sigma_u2=sigma_u2,
                hist=self.global_hist,
            )
        else:
            return reml_objective_from_summaries(
                self.Sxx, self.syy, beta,
                sigma2=sigma2, sigma_u2=sigma_u2,
                nj_sizes=(self.nj_sizes or []),
            )

    def gradient_logscale(self, Ainv: np.ndarray, sigma2: float, sigma_u2: float) -> np.ndarray:
        """∇ℓ wrt (log σ², log σᵤ²) from global summaries."""
        return reml_grad_logscale_from_summaries(
            q1=self.q1, q2=self.q2, T0=self.T0, T1=self.T1,
            B=self.B, Ainv=Ainv, p=self.p,
            sigma2=sigma2, sigma_u2=sigma_u2,
        )

    # ------------------------- variance update (line-search) ------------------------- #

    def update_variances(
        self,
        beta: np.ndarray,
        Ainv: np.ndarray,
        sigma2: float,
        sigma_u2: float,
        *,
        alpha: float = 1.0,
        c: float = 1e-4,
        max_backtracks: int = 8,
        min_step: float = 1e-6,
        lower_bound: float = 1e-12,
    ) -> Tuple[float, float, bool]:
        """
        One REML step on (σ², σᵤ²) using 2D backtracking line-search in log-space.

        Returns
        -------
        sigma2_new : float
        sigma_u2_new : float
        improved : bool
        """
        def ell_fn(phi: np.ndarray) -> float:
            s2 = float(np.exp(phi[0])); su2 = float(np.exp(phi[1]))
            return self.objective(beta, s2, su2)

        def grad_fn(phi: np.ndarray) -> np.ndarray:
            s2 = float(np.exp(phi[0])); su2 = float(np.exp(phi[1]))
            return self.gradient_logscale(Ainv, s2, su2)

        phi0 = np.array([np.log(sigma2), np.log(sigma_u2)], dtype=np.float64)
        phi_new, improved = backtracking_line_search_log2d(
            ell_fn, grad_fn, phi0,
            alpha=alpha, c=c,
            max_backtracks=max_backtracks,
            min_step=min_step,
            lower_bound=lower_bound,
        )
        return float(np.exp(phi_new[0])), float(np.exp(phi_new[1])), bool(improved)
