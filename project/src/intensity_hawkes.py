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

These correspond to `jump_self`, `jump_cross`, `beta` in `hawkes_simulator.py`
but are now passed through Softplus, matching the paper's nonlinear φ.

Relationship to ODE solver
--------------------------
The ODE solver (ode_solver_1d/2d.py) and closed-form approximations
(closed_form.py) continue to use the pure-exponential Λ(δ) Hamiltonian —
there is no closed-form Hamiltonian for the Hawkes-augmented intensity.
Strategy:
  • Use the ODE solver to compute optimal quotes (δ^b, δ^a).
  • Simulate fills using lambda_hawkes() below (Phase 1, option a).

References
----------
Lalor & Swishchuk (2025) — arXiv:2502.17417
Mei & Eisner (2017)      — Neural Hawkes Process (NeurIPS)
"""

from __future__ import annotations

import numpy as np

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
    # Compute log1p(exp(x)) only where safe to avoid overflow warnings
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
#  Hawkes intensity (vectorised in δ)
# ═══════════════════════════════════════════════════════════════════════════════

def lambda_hawkes(
    baseline: np.ndarray | float,
    excitation: np.ndarray | float,
) -> np.ndarray | float:
    """Hawkes intensity through Softplus nonlinearity.

    Parameters
    ----------
    baseline   : Λ(δ) = A·exp(−k·δ)  — quote-dependent baseline intensity
    excitation : y(t)                  — current excitation state (scalar or array)

    Returns
    -------
    λ = softplus( baseline + excitation )

    When excitation = 0, reduces exactly to softplus(Λ(δ)).
    For small Λ(δ) ≪ 1 and excitation = 0:
        softplus(Λ) ≈ Λ + ln(2)  [slight upward bias from the nonlinearity]
    Use `lambda_hawkes_approx_baseline` to recover the original Λ limit.

    Notes
    -----
    The paper uses φ = Softplus to ensure λ ≥ 0, consistent with Eq. (7) in
    Lalor & Swishchuk (2025) and Mei & Eisner (2017).
    """
    return softplus(np.asarray(baseline, dtype=float) +
                    np.asarray(excitation, dtype=float))


def lambda_hawkes_linear(
    baseline: np.ndarray | float,
    excitation: np.ndarray | float,
) -> np.ndarray | float:
    """Linear (non-Softplus) Hawkes intensity — for comparison / legacy use.

    λ = max(baseline + excitation, 0)

    Matches the behaviour of `hawkes_simulator.py` before this update.
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

    Exact for the continuous-time kernel μ(s) = α·exp(−β·s).
    """
    return y * np.exp(-beta * dt)


def update_excitation_bid_fill(
    y_bid: float,
    y_ask: float,
    alpha_self: float,
    alpha_cross: float,
) -> tuple[float, float]:
    """Update excitation states after a BID fill event.

    A bid fill corresponds to MB+ or MB0 in the paper's LOB event taxonomy.

    Self-excitation  : y_bid += α_self   (bid fill excites further bid fills)
    Cross-excitation : y_ask += α_cross  (bid fill excites ask fills — inventory
                                          rebalancing pressure)

    Returns
    -------
    (y_bid_new, y_ask_new)
    """
    return y_bid + alpha_self, y_ask + alpha_cross


def update_excitation_ask_fill(
    y_bid: float,
    y_ask: float,
    alpha_self: float,
    alpha_cross: float,
) -> tuple[float, float]:
    """Update excitation states after an ASK fill event.

    A ask fill corresponds to MS- or MS0 in the paper's LOB event taxonomy.

    Self-excitation  : y_ask += α_self
    Cross-excitation : y_bid += α_cross
    """
    return y_bid + alpha_cross, y_ask + alpha_self


# ═══════════════════════════════════════════════════════════════════════════════
#  Fill probability (exact Poisson)
# ═══════════════════════════════════════════════════════════════════════════════

def fill_prob_from_intensity(lam: np.ndarray | float, dt: float) -> np.ndarray | float:
    """Probability of at least one fill in [t, t+dt] for intensity λ.

    P(fill) = 1 − exp(−λ · dt)

    More accurate than the first-order approximation λ·dt at high intensities,
    consistent with Ogata's thinning construction used in the paper.
    """
    return 1.0 - np.exp(-np.asarray(lam, dtype=float) * dt)


# ═══════════════════════════════════════════════════════════════════════════════
#  High-level per-step update (convenience wrapper for simulator integration)
# ═══════════════════════════════════════════════════════════════════════════════

class HawkesState:
    """Mutable container for one simulation path's Hawkes excitation states.

    Designed for integration into a discrete-time simulator loop:

        state = HawkesState(hawkes_cfg)
        for t in range(N):
            lam_b = state.lambda_bid(mu_bid)
            lam_a = state.lambda_ask(mu_ask)
            prob_b = fill_prob_from_intensity(lam_b, dt)
            prob_a = fill_prob_from_intensity(lam_a, dt)
            fill_b = rng.random() < prob_b
            fill_a = rng.random() < prob_a
            state.step(dt, fill_b, fill_a)

    Parameters
    ----------
    cfg : dict with keys
        beta        — decay rate β
        alpha_self  — self-excitation jump α_ii
        alpha_cross — cross-excitation jump α_ij

    Backward-compatible aliases (hawkes_simulator.py convention):
        beta, jump_self, jump_cross   are accepted as alternates.
    """

    def __init__(self, cfg: dict, y_bid0: float = 0.0, y_ask0: float = 0.0):
        self.beta        = cfg.get("beta", cfg.get("beta", 1.0))
        self.alpha_self  = cfg.get("alpha_self",  cfg.get("jump_self",  0.0))
        self.alpha_cross = cfg.get("alpha_cross", cfg.get("jump_cross", 0.0))
        self.y_bid = y_bid0
        self.y_ask = y_ask0

    # ── intensity accessors ───────────────────────────────────────────────────

    def lambda_bid(self, mu_bid: float) -> float:
        """λ_bid = softplus(Λ(δ^b) + y_bid)."""
        return float(lambda_hawkes(mu_bid, self.y_bid))

    def lambda_ask(self, mu_ask: float) -> float:
        """λ_ask = softplus(Λ(δ^a) + y_ask)."""
        return float(lambda_hawkes(mu_ask, self.y_ask))

    # ── one time-step update ──────────────────────────────────────────────────

    def step(self, dt: float, fill_bid: bool, fill_ask: bool) -> None:
        """Advance excitation states by one time step dt.

        Order of operations (matches paper's discrete recursion):
          1. Apply exponential decay.
          2. Add jumps for any fills that occurred this step.
        """
        # 1. decay
        self.y_bid = decay_excitation(self.y_bid, self.beta, dt)
        self.y_ask = decay_excitation(self.y_ask, self.beta, dt)

        # 2. jumps
        if fill_bid:
            self.y_bid, self.y_ask = update_excitation_bid_fill(
                self.y_bid, self.y_ask, self.alpha_self, self.alpha_cross
            )
        if fill_ask:
            self.y_bid, self.y_ask = update_excitation_ask_fill(
                self.y_bid, self.y_ask, self.alpha_self, self.alpha_cross
            )

    def reset(self) -> None:
        """Reset excitation states to zero (start of new simulation path)."""
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
