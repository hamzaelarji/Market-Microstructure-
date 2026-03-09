"""Intensity functions and Hamiltonian for exponential Λ(δ) = A·exp(−k·δ).

Multi‑asset convention (Section 5 of Guéant 2017):
  ξ  = γ (Model A) or 0 (Model B)
  ξΔ = ξ·Δ   — the product that enters the Hamiltonian

Definitions (exponential case, closed‑form):
  C(ξΔ)     = (1 + ξΔ/k)^{−(k/(ξΔ) + 1)}         ξΔ > 0
              e^{−1}                                  ξΔ = 0

  H_ξ(p)    = (A·Δ/k) · C(ξΔ) · exp(−k·p)

  δ*(p)     = p + (1/(ξΔ)) · ln(1 + ξΔ/k)          ξΔ > 0
              p + 1/k                                 ξΔ = 0
"""

import numpy as np

EPS = 1e-12  # threshold below which ξΔ is treated as 0


# ─── elementary ──────────────────────────────────────────────────────
def Lambda(delta, A, k):
    """Λ(δ) = A·exp(−k·δ)."""
    return A * np.exp(-k * np.asarray(delta, dtype=float))


# ─── coefficient C ───────────────────────────────────────────────────
def C_coeff(xi_Delta, k):
    """C(ξΔ) — scalar only."""
    if abs(xi_Delta) < EPS:
        return np.exp(-1.0)
    u = xi_Delta / k
    return (1.0 + u) ** (-(k / xi_Delta + 1.0))


# ─── Hamiltonian and derivatives ─────────────────────────────────────
def H_val(p, xi, A, k, Delta):
    """H_ξ(p) = (AΔ/k)·C(ξΔ)·exp(−k·p).  Vectorised in p."""
    C = C_coeff(xi * Delta, k)
    return (A * Delta / k) * C * np.exp(-k * np.asarray(p, dtype=float))


def H_prime(p, xi, A, k, Delta):
    """H'_ξ(p) = −k · H_ξ(p)."""
    return -k * H_val(p, xi, A, k, Delta)


def H_second(p, xi, A, k, Delta):
    """H''_ξ(p) = k² · H_ξ(p)."""
    return k ** 2 * H_val(p, xi, A, k, Delta)


# ─── optimal spread ─────────────────────────────────────────────────
def delta_star(p, xi, k, Delta):
    """δ*(p) — the distance‑to‑mid that realises the sup in H_ξ(p).
    Vectorised in p.
    """
    p = np.asarray(p, dtype=float)
    xi_Delta = xi * Delta
    if abs(xi_Delta) < EPS:
        return p + 1.0 / k
    return p + (1.0 / xi_Delta) * np.log(1.0 + xi_Delta / k)


# ─── fill probability — source unique de vérité ──────────────────────
def fill_prob(lam: float, dt: float) -> float:
    """P(≥1 fill dans [t, t+dt]) = 1 − exp(−λ·dt).

    Plus précis que l'approximation linéaire λ·dt aux fortes intensités.
    """
    return 1.0 - np.exp(-max(float(lam), 0.0) * dt)
