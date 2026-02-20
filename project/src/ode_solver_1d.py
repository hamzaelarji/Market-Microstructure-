"""Single‑asset ODE solver for optimal market making (Guéant 2017, Sections 3–4).

Model A (ξ = γ):  linear tridiagonal system via v‑transform  v_n = exp(−k·θ_n / Δ)
Model B (ξ = 0):  Newton iteration on the nonlinear implicit Euler

Both solvers step backward in time from θ(T) = −ℓ(|q|) to t = 0.

Convention: inventory in lots  n ∈ {−Q, …, +Q},  notional  q = n·Δ.
The ODE (multi‑asset convention, Eq. 5.13 specialised to d = 1):

  ∂_t θ_n + ½ γ σ² (nΔ)²
    − 𝟙_{n<Q}  H_ξ( (θ_n − θ_{n+1}) / Δ )
    − 𝟙_{n>−Q} H_ξ( (θ_n − θ_{n−1}) / Δ )  = 0

Terminal condition:  θ_n(T) = −ℓ(|n Δ|).
"""

import numpy as np
from scipy.linalg import solve_banded
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.intensity import C_coeff, H_val, H_prime, delta_star


# ═══════════════════════════════════════════════════════════════════════
#  MODEL A  —  linear tridiagonal solver
# ═══════════════════════════════════════════════════════════════════════
def solve_model_a(params, gamma, T, N_t=7200, ell_func=None):
    """Solve Model A via the linear v‑transformation.

    The exponential utility u = −exp(−γ(x + qS + θ)) leads to:
        v_n := exp(−k θ_n / Δ)
    satisfying the LINEAR ODE (backward time τ = T − t):
        dv_n/dτ = −½ k γ σ² n² Δ · v_n  +  A C_{ξΔ} · (v_{n+1} + v_{n−1})

    Implicit Euler in τ gives a constant tridiagonal system per step.

    Returns
    -------
    dict with keys: theta, delta_bid, delta_ask, times, lots, params, gamma, xi
    """
    sigma = params["sigma"]
    A     = params["A"]
    k     = params["k"]
    Delta = params["Delta"]
    Q     = params["Q"]

    xi = gamma
    dt = T / N_t
    lots = np.arange(-Q, Q + 1, dtype=float)   # −4 … +4
    N = len(lots)

    # Coefficients
    xi_Delta = xi * Delta
    C = C_coeff(xi_Delta, k)
    alpha = 0.5 * k * gamma * sigma**2 * Delta   # coeff on n² in diagonal
    beta  = A * C                                  # off‑diagonal coupling

    # ── terminal condition ──
    if ell_func is None:
        v = np.ones(N)
    else:
        # θ(T) = −ℓ(|q|),  v = exp(+k θ / Δ) = exp(−k ℓ / Δ)
        v = np.array([np.exp(-k * ell_func(abs(n * Delta)) / Delta) for n in lots])

    # ── build CONSTANT banded matrix  (I + dt·M) ──
    #   M_nn   =  α n²        (inventory penalty)
    #   M_n,n±1 = −β           (execution coupling)
    ab = np.zeros((3, N))
    ab[1, :] = 1.0 + dt * alpha * lots**2          # main diagonal
    ab[0, 1:]  = -dt * beta                         # upper  (stored shifted)
    ab[2, :-1] = -dt * beta                         # lower  (stored shifted)

    # ── time‑stepping ──
    # Store v so we can recover θ and quotes at every time
    v_hist = np.zeros((N_t + 1, N))
    v_hist[0] = v.copy()                            # backward‑time index 0 = physical time T

    for m in range(N_t):
        v = solve_banded((1, 1), ab, v)
        v = np.maximum(v, 1e-300)                   # numerical guard
        v_hist[m + 1] = v

    # ── convert to θ and reverse time ──
    # v_hist[m] → physical time T − m dt
    # v = exp(+k θ / Δ)  ⟹  θ = +(Δ/k) ln(v)
    theta_bwd = (Delta / k) * np.log(v_hist)       # shape (N_t+1, N)
    theta = theta_bwd[::-1]                          # theta[i] → t_i = i·dt
    times = np.linspace(0.0, T, N_t + 1)

    # ── extract quotes ──
    delta_bid, delta_ask = _extract_quotes(theta, lots, xi, k, Delta, Q)

    return dict(theta=theta, delta_bid=delta_bid, delta_ask=delta_ask,
                times=times, lots=lots, params=params, gamma=gamma, xi=xi)


# ═══════════════════════════════════════════════════════════════════════
#  MODEL B  —  Newton on implicit Euler
# ═══════════════════════════════════════════════════════════════════════
def solve_model_b(params, gamma, T, N_t=7200, ell_func=None):
    """Solve Model B (ξ = 0) via Newton iteration on implicit backward Euler.

    Returns the same dict structure as solve_model_a.
    """
    sigma = params["sigma"]
    A     = params["A"]
    k     = params["k"]
    Delta = params["Delta"]
    Q     = params["Q"]

    xi = 0.0
    dt = T / N_t
    lots = np.arange(-Q, Q + 1, dtype=float)
    N = len(lots)

    # ── terminal condition ──
    if ell_func is None:
        theta_old = np.zeros(N)
    else:
        theta_old = np.array([-ell_func(abs(n * Delta)) for n in lots])

    theta_bwd = np.zeros((N_t + 1, N))
    theta_bwd[0] = theta_old.copy()

    for m in range(N_t):
        theta_new = theta_old.copy()         # initial guess = previous step

        for _ in range(12):                  # Newton iterations
            G, J_main, J_upper, J_lower = \
                _newton_residual(theta_new, theta_old, dt, lots, gamma, sigma, A, k, Delta, Q, xi)

            # Solve tridiagonal  J · correction = −G
            ab_J = np.zeros((3, N))
            ab_J[1, :] = J_main
            ab_J[0, 1:] = J_upper
            ab_J[2, :-1] = J_lower
            corr = solve_banded((1, 1), ab_J, -G)
            theta_new += corr

            if np.max(np.abs(corr)) < 1e-15:
                break

        theta_old = theta_new.copy()
        theta_bwd[m + 1] = theta_old

    # ── reverse time ──
    theta = theta_bwd[::-1]
    times = np.linspace(0.0, T, N_t + 1)
    delta_bid, delta_ask = _extract_quotes(theta, lots, xi, k, Delta, Q)

    return dict(theta=theta, delta_bid=delta_bid, delta_ask=delta_ask,
                times=times, lots=lots, params=params, gamma=gamma, xi=xi)


# ═══════════════════════════════════════════════════════════════════════
#  GENERAL SOLVER  — Newton, works for any ξ ≥ 0
# ═══════════════════════════════════════════════════════════════════════
def solve_general(params, gamma, T, xi, N_t=7200, ell_func=None):
    """General solver (Newton) for arbitrary ξ.  Use for Model A or B."""
    sigma = params["sigma"]
    A     = params["A"]
    k     = params["k"]
    Delta = params["Delta"]
    Q     = params["Q"]

    dt = T / N_t
    lots = np.arange(-Q, Q + 1, dtype=float)
    N = len(lots)

    if ell_func is None:
        theta_old = np.zeros(N)
    else:
        theta_old = np.array([-ell_func(abs(n * Delta)) for n in lots])

    theta_bwd = np.zeros((N_t + 1, N))
    theta_bwd[0] = theta_old.copy()

    for m in range(N_t):
        theta_new = theta_old.copy()
        for _ in range(12):
            G, J_main, J_upper, J_lower = \
                _newton_residual(theta_new, theta_old, dt, lots, gamma, sigma, A, k, Delta, Q, xi)
            ab_J = np.zeros((3, N))
            ab_J[1, :] = J_main
            ab_J[0, 1:] = J_upper
            ab_J[2, :-1] = J_lower
            corr = solve_banded((1, 1), ab_J, -G)
            theta_new += corr
            if np.max(np.abs(corr)) < 1e-15:
                break
        theta_old = theta_new.copy()
        theta_bwd[m + 1] = theta_old

    theta = theta_bwd[::-1]
    times = np.linspace(0.0, T, N_t + 1)
    delta_bid, delta_ask = _extract_quotes(theta, lots, xi, k, Delta, Q)

    return dict(theta=theta, delta_bid=delta_bid, delta_ask=delta_ask,
                times=times, lots=lots, params=params, gamma=gamma, xi=xi)


# ═══════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════
def _newton_residual(theta_new, theta_old, dt, lots, gamma, sigma, A, k, Delta, Q, xi):
    """Compute residual G and tridiagonal Jacobian for implicit Euler.

    G_n = θ_new_n − θ_old_n + dt·F_n(θ_new)
    F_n = ½ γ σ² (nΔ)²  −  𝟙_{n<Q} H(p_bid)  −  𝟙_{n>−Q} H(p_ask)
    """
    N = len(lots)
    G = np.zeros(N)
    J_main  = np.zeros(N)
    J_upper = np.zeros(N - 1)
    J_lower = np.zeros(N - 1)

    for i in range(N):
        n = lots[i]
        q = n * Delta
        inv_pen = 0.5 * gamma * sigma**2 * q**2

        Hb = 0.0;  dHb_dn = 0.0;  dHb_dnp = 0.0
        Ha = 0.0;  dHa_dn = 0.0;  dHa_dnm = 0.0

        # bid side  (n < Q  →  fill raises inventory to n+1)
        if n < Q and i + 1 < N:
            pb = (theta_new[i] - theta_new[i + 1]) / Delta
            Hb = H_val(pb, xi, A, k, Delta)
            Hpb = -k * Hb                           # H'(p_bid)
            dHb_dn  =  Hpb / Delta                  # ∂H/∂θ_n
            dHb_dnp = -Hpb / Delta                   # ∂H/∂θ_{n+1}

        # ask side  (n > −Q  →  fill lowers inventory to n−1)
        if n > -Q and i - 1 >= 0:
            pa = (theta_new[i] - theta_new[i - 1]) / Delta
            Ha = H_val(pa, xi, A, k, Delta)
            Hpa = -k * Ha
            dHa_dn  =  Hpa / Delta
            dHa_dnm = -Hpa / Delta

        # Residual
        G[i] = theta_new[i] - theta_old[i] + dt * (inv_pen - Hb - Ha)

        # Jacobian diagonal:  ∂G_n/∂θ_n = 1 + dt·(−dHb/dθ_n − dHa/dθ_n)
        J_main[i] = 1.0 + dt * (-dHb_dn - dHa_dn)

        # Off‑diagonals
        if n < Q and i + 1 < N:
            J_upper[i] = dt * (-dHb_dnp)            # ∂G_n/∂θ_{n+1}
        if n > -Q and i - 1 >= 0:
            J_lower[i - 1] = dt * (-dHa_dnm)        # ∂G_n/∂θ_{n−1}

    return G, J_main, J_upper, J_lower


def _extract_quotes(theta, lots, xi, k, Delta, Q):
    """From θ array (T_steps × N_lots), compute δ^b and δ^a."""
    N_t_plus_1, N = theta.shape
    delta_bid = np.full((N_t_plus_1, N), np.nan)
    delta_ask = np.full((N_t_plus_1, N), np.nan)

    for i in range(N):
        n = lots[i]
        if n < Q and i + 1 < N:
            pb = (theta[:, i] - theta[:, i + 1]) / Delta
            delta_bid[:, i] = delta_star(pb, xi, k, Delta)
        if n > -Q and i - 1 >= 0:
            pa = (theta[:, i] - theta[:, i - 1]) / Delta
            delta_ask[:, i] = delta_star(pa, xi, k, Delta)

    return delta_bid, delta_ask