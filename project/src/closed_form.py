"""Closed‑form approximations for optimal quotes (Guéant 2017, Section 4).

Generalised Guéant–Lehalle–Fernandez‑Tapia formulas.
Approximation valid far from T, exponential intensities, symmetric Λ^b = Λ^a.

Multi‑asset convention:  ξ = γ (Model A) or 0 (Model B), ξΔ = ξ·Δ.

  δ^b_approx(n) = δ_static  +  ω · (2n + 1) Δ / 2
  δ^a_approx(n) = δ_static  −  ω · (2n − 1) Δ / 2

where
  δ_static  =  (1/(ξΔ)) ln(1 + ξΔ/k)        if ξ > 0
                1/k                             if ξ = 0

  ω = √( γ σ² / (2 H''_ξ(0)) )
    = √( γ σ² / (2 A Δ k C(ξΔ)) )

  spread = δ^b + δ^a = 2·δ_static + ω · Δ
  skew   = δ^b − δ^a = 2n · ω · Δ
"""

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.intensity import C_coeff


def approx_quotes(n, params, gamma, xi):
    """Closed‑form approximation for a single lot index n.

    Parameters
    ----------
    n : int or array  — inventory in lots
    params : dict with sigma, A, k, Delta, Q
    gamma : float — risk aversion
    xi : float — 0 for Model B, γ for Model A

    Returns
    -------
    delta_bid, delta_ask : same shape as n
    """
    sigma = params["sigma"]
    A     = params["A"]
    k     = params["k"]
    Delta = params["Delta"]

    n = np.asarray(n, dtype=float)
    xi_Delta = xi * Delta

    # static (half‑spread baseline)
    if abs(xi_Delta) < 1e-12:
        d_static = 1.0 / k
    else:
        d_static = (1.0 / xi_Delta) * np.log(1.0 + xi_Delta / k)

    # dynamic slope
    C = C_coeff(xi_Delta, k)
    omega = np.sqrt(gamma * sigma**2 / (2.0 * A * Delta * k * C))

    delta_bid = d_static + omega * (2 * n + 1) * Delta / 2.0
    delta_ask = d_static - omega * (2 * n - 1) * Delta / 2.0

    return delta_bid, delta_ask


def approx_spread(n, params, gamma, xi):
    """Spread ≈ 2·δ_static + ω·Δ  (constant in n for exponential Λ)."""
    db, da = approx_quotes(n, params, gamma, xi)
    return db + da


def approx_skew(n, params, gamma, xi):
    """Skew ≈ 2n · ω · Δ  (linear in n for exponential Λ)."""
    db, da = approx_quotes(n, params, gamma, xi)
    return db - da