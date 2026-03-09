"""Hawkes-based intensity for optimal market making (Phase 1 — Parametric).

Inspired by Lalor & Swishchuk (2025), "Event-Based Limit Order Book
Simulation under a Neural Hawkes Process: Application in Market-Making",
arXiv:2502.17417.

Framework
---------
The paper models 12 LOB event types via a nonlinear Multivariate Hawkes
Process (MVHP):

    λ_i(t) = φ_i( λ̄_i + ∫_{(0,t)} Σ_j α_ij · exp(−β_ij(t−s)) dN_j(s) )

where φ_i is the Softplus nonlinearity: φ(x) = ln(1 + exp(x)).

For our **single-asset** Guéant-style market-making context we reduce this
to two event streams — bid fills and ask fills — and keep the exponential
Λ(δ) = A·exp(−k·δ) as the quote-dependent baseline:

    λ_bid(t) = softplus( Λ(δ^b)   + y_bid(t) )
    λ_ask(t) = softplus( Λ(δ^a)   + y_ask(t) )

where the excitation states evolve via the discrete-time recursion (exact
for piecewise-constant δ over one time step dt):

    Decay step  :  y ← y · exp(−β · dt)
    Jump on fill:  y_bid ← y_bid + α_self    (self-excitation from bid fill)
                   y_ask ← y_ask + α_cross   (cross-excitation from bid fill)
    (symmetrically for ask fills)

Parameters (hawkes_cfg dict)
----------------------------
beta        : float  — exponential decay rate of the kernel (β in the paper)
alpha_self  : float  — jump size for self-excitation  (α_ii)
alpha_cross : float  — jump size for cross-excitation (α_ij, i≠j)

References
----------
Lalor & Swishchuk (2025) — arXiv:2502.17417
Mei & Eisner (2017)      — Neural Hawkes Process (NeurIPS)
"""

from __future__ import annotations

import numpy as np
from market_making.core.intensity import fill_prob

# ── numerical guard against overflow in exp ──────────────────────────────────
_SOFTPLUS_CLIP = 50.0   # for |x| > 50 use asymptotic form


# ═══════════════════════════════════════════════════════════════════════════════
#  Core nonlinear transfer function
# ═══════════════════════════════════════════════════════════════════════════════

def softplus(x: np.ndarray | float) -> np.ndarray | float:
    """Softplus activation: φ(x) = ln(1 + exp(x)).

    Numerically stable implementation:
      • x >  50 :  φ(x) ≈ x          (linear regime)
      • x < −50 :  φ(x) ≈ exp(x) ≈ 0 (near-zero regime)
      • else     :  exact formula
    """
    x = np.asarray(x, dtype=float)
    safe = (x > -_SOFTPLUS_CLIP) & (x <= _SOFTPLUS_CLIP)
    out = np.empty_like(x)
    out[x > _SOFTPLUS_CLIP]  = x[x > _SOFTPLUS_CLIP]
    out[x < -_SOFTPLUS_CLIP] = np.exp(x[x < -_SOFTPLUS_CLIP])
    out[safe]                = np.log1p(np.exp(x[safe]))
    return out


def softplus_deriv(x: np.ndarray | float) -> np.ndarray | float:
    """φ'(x) = sigmoid(x) = 1 / (1 + exp(−x))."""
    x = np.asarray(x, dtype=float)
    return np.where(
        x > _SOFTPLUS_CLIP,
        1.0,
        np.where(
            x < -_SOFTPLUS_CLIP,
            np.exp(x),
            1.0 / (1.0 + np.exp(-x)),
        )
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Hawkes intensity (vectorised en δ)
# ═══════════════════════════════════════════════════════════════════════════════

def lambda_hawkes(
    baseline: np.ndarray | float,
    excitation: np.ndarray | float,
) -> np.ndarray | float:
    """Hawkes intensity through Softplus nonlinearity.

    λ = softplus( baseline + excitation )

    When excitation = 0, reduces exactly to softplus(Λ(δ)).
    """
    return softplus(np.asarray(baseline, dtype=float) +
                    np.asarray(excitation, dtype=float))


def lambda_hawkes_linear(
    baseline: np.ndarray | float,
    excitation: np.ndarray | float,
) -> np.ndarray | float:
    """Linear (non-Softplus) Hawkes intensity — for comparison / legacy use.

    λ = max(baseline + excitation, 0)

    Prefer `lambda_hawkes` (Softplus) for the paper-consistent formulation.
    """
    return np.maximum(
        np.asarray(baseline, dtype=float) + np.asarray(excitation, dtype=float),
        0.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Excitation state helpers
# ═══════════════════════════════════════════════════════════════════════════════

def decay_excitation(y: float, beta: float, dt: float) -> float:
    """Exponential decay of the excitation state over one time step dt.

    y(t + dt) = y(t) · exp(−β · dt)
    """
    return y * np.exp(-beta * dt)


def update_excitation_bid_fill(
    y_bid: float,
    y_ask: float,
    alpha_self: float,
    alpha_cross: float,
) -> tuple[float, float]:
    """Update excitation states after a BID fill event."""
    return y_bid + alpha_self, y_ask + alpha_cross


def update_excitation_ask_fill(
    y_bid: float,
    y_ask: float,
    alpha_self: float,
    alpha_cross: float,
) -> tuple[float, float]:
    """Update excitation states after an ASK fill event."""
    return y_bid + alpha_cross, y_ask + alpha_self


# ═══════════════════════════════════════════════════════════════════════════════
#  Fill probability — re-exportée depuis intensity pour la compatibilité
# ═══════════════════════════════════════════════════════════════════════════════

# Alias pour la compatibilité avec l'ancien nom
fill_prob_from_intensity = fill_prob


# ═══════════════════════════════════════════════════════════════════════════════
#  HawkesState — conteneur mutable pour un chemin de simulation
# ═══════════════════════════════════════════════════════════════════════════════

class HawkesState:
    """Mutable container for one simulation path's Hawkes excitation states.

    Usage:
        state = HawkesState(hawkes_cfg)
        for t in range(N):
            lam_b = state.lambda_bid(mu_bid)
            lam_a = state.lambda_ask(mu_ask)
            fill_b = rng.random() < fill_prob(lam_b, dt)
            fill_a = rng.random() < fill_prob(lam_a, dt)
            state.step(dt, fill_b, fill_a)

    Parameters
    ----------
    cfg : dict with keys
        beta        — decay rate β
        alpha_self  — self-excitation jump α_ii
        alpha_cross — cross-excitation jump α_ij
        (also accepts legacy aliases jump_self, jump_cross)
    """

    def __init__(self, cfg: dict, y_bid0: float = 0.0, y_ask0: float = 0.0):
        self.beta        = cfg.get("beta", 1.0)
        self.alpha_self  = cfg.get("alpha_self",  cfg.get("jump_self",  0.0))
        self.alpha_cross = cfg.get("alpha_cross", cfg.get("jump_cross", 0.0))
        self.y_bid = y_bid0
        self.y_ask = y_ask0

    def lambda_bid(self, mu_bid: float) -> float:
        """λ_bid = softplus(Λ(δ^b) + y_bid)."""
        return float(lambda_hawkes(mu_bid, self.y_bid))

    def lambda_ask(self, mu_ask: float) -> float:
        """λ_ask = softplus(Λ(δ^a) + y_ask)."""
        return float(lambda_hawkes(mu_ask, self.y_ask))

    def step(self, dt: float, fill_bid: bool, fill_ask: bool) -> None:
        """Advance excitation states by one time step dt.

        Order: 1. exponential decay, 2. jumps from fills.
        """
        self.y_bid = decay_excitation(self.y_bid, self.beta, dt)
        self.y_ask = decay_excitation(self.y_ask, self.beta, dt)

        if fill_bid:
            self.y_bid, self.y_ask = update_excitation_bid_fill(
                self.y_bid, self.y_ask, self.alpha_self, self.alpha_cross
            )
        if fill_ask:
            self.y_bid, self.y_ask = update_excitation_ask_fill(
                self.y_bid, self.y_ask, self.alpha_self, self.alpha_cross
            )

    def reset(self) -> None:
        """Reset excitation states to zero."""
        self.y_bid = 0.0
        self.y_ask = 0.0

    def __repr__(self) -> str:
        return (f"HawkesState(β={self.beta}, α_self={self.alpha_self}, "
                f"α_cross={self.alpha_cross}, "
                f"y_bid={self.y_bid:.4f}, y_ask={self.y_ask:.4f})")


# ═══════════════════════════════════════════════════════════════════════════════
#  Default Hawkes configuration
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_HAWKES_CFG = {
    "beta":        10.0,   # fast mean-reversion (≈ 0.1 s half-life)
    "alpha_self":   2.0,   # moderate self-excitation
    "alpha_cross":  0.5,   # weaker cross-excitation
}
