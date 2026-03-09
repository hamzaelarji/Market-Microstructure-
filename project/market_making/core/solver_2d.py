"""Multi‑asset (d = 2) ODE solver for optimal market making (Guéant 2017, Section 5).

Solves Eq. (5.13) specialised to d = 2 assets with exponential intensities.
The solver uses Newton iteration on implicit backward Euler with sparse Jacobian.

Convention:
  Inventory in lots  n_i ∈ {−Q_i, …, +Q_i}  for asset i ∈ {1, 2}.
  Notional  q_i = n_i · Δ_i.

ODE (backward in time):
  ∂_t θ(n) + ½ γ q^T Σ q
      − Σ_i 𝟙_{n_i < Q_i}  H_ξ^i( (θ(n) − θ(n + e_i)) / Δ_i )
      − Σ_i 𝟙_{n_i > −Q_i} H_ξ^i( (θ(n) − θ(n − e_i)) / Δ_i )  = 0

Terminal condition:  θ(n)(T) = −ℓ(n_1 Δ_1, n_2 Δ_2).
"""

import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from market_making.core.intensity import H_val, H_prime, delta_star


# ═══════════════════════════════════════════════════════════════════════
#  Grid helpers
# ═══════════════════════════════════════════════════════════════════════

def _build_grid(Q1, Q2):
    """Build the 2D lot grid and index mapping.

    Returns
    -------
    grid : ndarray of shape (N, 2) — each row is (n1, n2) in lots
    idx  : dict mapping (n1, n2) → flat index
    N    : total number of grid points
    """
    lots1 = np.arange(-Q1, Q1 + 1)
    lots2 = np.arange(-Q2, Q2 + 1)
    grid = []
    idx = {}
    k = 0
    for n1 in lots1:
        for n2 in lots2:
            grid.append((n1, n2))
            idx[(n1, n2)] = k
            k += 1
    return np.array(grid), idx, k


# ═══════════════════════════════════════════════════════════════════════
#  Main solver
# ═══════════════════════════════════════════════════════════════════════

def solve_2d(params1, params2, gamma, rho, T, xi, N_t=7200, ell_func=None):
    """Solve the 2‑asset ODE (5.13) via Newton on implicit backward Euler.

    Parameters
    ----------
    params1, params2 : dicts with keys sigma, A, k, Delta, Q  (per‑asset)
    gamma : float — risk aversion
    rho   : float — correlation between the two assets
    T     : float — time horizon (seconds)
    xi    : float — 0 for Model B, γ for Model A
    N_t   : int   — number of time steps
    ell_func : callable(q1, q2) → penalty, or None for ℓ = 0

    Returns
    -------
    dict with keys:
        theta          : ndarray (N_t+1, N_grid) — value function on the grid
        delta_bid_1/2  : ndarray (N_t+1, N_grid) — bid quotes per asset
        delta_ask_1/2  : ndarray (N_t+1, N_grid) — ask quotes per asset
        times          : ndarray (N_t+1,)
        grid           : ndarray (N_grid, 2) — lot indices
        idx            : dict  (n1, n2) → flat index
        params1, params2, gamma, rho, xi
    """
    s1, A1, k1, D1, Q1 = (params1[k] for k in ("sigma", "A", "k", "Delta", "Q"))
    s2, A2, k2, D2, Q2 = (params2[k] for k in ("sigma", "A", "k", "Delta", "Q"))

    dt = T / N_t
    grid, idx, N = _build_grid(Q1, Q2)

    Sig = np.array([[s1**2, rho * s1 * s2],
                    [rho * s1 * s2, s2**2]])

    theta_old = np.zeros(N)
    if ell_func is not None:
        for j, (n1, n2) in enumerate(grid):
            theta_old[j] = -ell_func(n1 * D1, n2 * D2)

    theta_bwd = np.zeros((N_t + 1, N))
    theta_bwd[0] = theta_old.copy()

    for m in range(N_t):
        theta_new = theta_old.copy()

        for newton_iter in range(12):
            G, J = _residual_and_jacobian(
                theta_new, theta_old, dt, grid, idx, N,
                gamma, Sig, xi,
                A1, k1, D1, Q1,
                A2, k2, D2, Q2)

            J_csc = csc_matrix(J)
            corr = spsolve(J_csc, -G)
            theta_new += corr

            if np.max(np.abs(corr)) < 1e-14:
                break

        theta_old = theta_new.copy()
        theta_bwd[m + 1] = theta_old

        if (m + 1) % max(1, N_t // 10) == 0:
            print(f"  2D solver: step {m+1}/{N_t}  "
                  f"(Newton iters={newton_iter+1}, |corr|={np.max(np.abs(corr)):.2e})")

    theta = theta_bwd[::-1]
    times = np.linspace(0.0, T, N_t + 1)

    db1, da1, db2, da2 = _extract_quotes_2d(
        theta, grid, idx, N, xi,
        k1, D1, Q1, k2, D2, Q2)

    return dict(
        theta=theta,
        delta_bid_1=db1, delta_ask_1=da1,
        delta_bid_2=db2, delta_ask_2=da2,
        times=times, grid=grid, idx=idx,
        params1=params1, params2=params2,
        gamma=gamma, rho=rho, xi=xi)


# ═══════════════════════════════════════════════════════════════════════
#  Newton residual & Jacobian
# ═══════════════════════════════════════════════════════════════════════

def _residual_and_jacobian(theta_new, theta_old, dt, grid, idx, N,
                           gamma, Sig, xi,
                           A1, k1, D1, Q1,
                           A2, k2, D2, Q2):
    """Compute G and sparse Jacobian J for the 2D implicit Euler step."""
    G = np.zeros(N)
    J = lil_matrix((N, N))

    for j in range(N):
        n1, n2 = grid[j]
        q = np.array([n1 * D1, n2 * D2], dtype=float)
        inv_pen = 0.5 * gamma * q @ Sig @ q

        H_total = 0.0
        dH_dj = 0.0

        if n1 < Q1:
            nb = (n1 + 1, n2)
            if nb in idx:
                jb = idx[nb]
                p = (theta_new[j] - theta_new[jb]) / D1
                Hb = H_val(p, xi, A1, k1, D1)
                Hp = -k1 * Hb
                H_total += Hb
                dH_dj += Hp / D1
                J[j, jb] += dt * (Hp / D1)

        if n1 > -Q1:
            na = (n1 - 1, n2)
            if na in idx:
                ja = idx[na]
                p = (theta_new[j] - theta_new[ja]) / D1
                Ha = H_val(p, xi, A1, k1, D1)
                Hp = -k1 * Ha
                H_total += Ha
                dH_dj += Hp / D1
                J[j, ja] += dt * (Hp / D1)

        if n2 < Q2:
            nb = (n1, n2 + 1)
            if nb in idx:
                jb = idx[nb]
                p = (theta_new[j] - theta_new[jb]) / D2
                Hb = H_val(p, xi, A2, k2, D2)
                Hp = -k2 * Hb
                H_total += Hb
                dH_dj += Hp / D2
                J[j, jb] += dt * (Hp / D2)

        if n2 > -Q2:
            na = (n1, n2 - 1)
            if na in idx:
                ja = idx[na]
                p = (theta_new[j] - theta_new[ja]) / D2
                Ha = H_val(p, xi, A2, k2, D2)
                Hp = -k2 * Ha
                H_total += Ha
                dH_dj += Hp / D2
                J[j, ja] += dt * (Hp / D2)

        G[j] = theta_new[j] - theta_old[j] + dt * (inv_pen - H_total)
        J[j, j] = 1.0 + dt * (-dH_dj)

    return G, J


# ═══════════════════════════════════════════════════════════════════════
#  Quote extraction
# ═══════════════════════════════════════════════════════════════════════

def _extract_quotes_2d(theta, grid, idx, N, xi,
                       k1, D1, Q1, k2, D2, Q2):
    """From θ (N_t+1 × N_grid), compute δ^b and δ^a for each asset."""
    Nt1 = theta.shape[0]
    db1 = np.full((Nt1, N), np.nan)
    da1 = np.full((Nt1, N), np.nan)
    db2 = np.full((Nt1, N), np.nan)
    da2 = np.full((Nt1, N), np.nan)

    for j, (n1, n2) in enumerate(grid):
        if n1 < Q1:
            nb = (n1 + 1, n2)
            if nb in idx:
                p = (theta[:, j] - theta[:, idx[nb]]) / D1
                db1[:, j] = delta_star(p, xi, k1, D1)
        if n1 > -Q1:
            na = (n1 - 1, n2)
            if na in idx:
                p = (theta[:, j] - theta[:, idx[na]]) / D1
                da1[:, j] = delta_star(p, xi, k1, D1)
        if n2 < Q2:
            nb = (n1, n2 + 1)
            if nb in idx:
                p = (theta[:, j] - theta[:, idx[nb]]) / D2
                db2[:, j] = delta_star(p, xi, k2, D2)
        if n2 > -Q2:
            na = (n1, n2 - 1)
            if na in idx:
                p = (theta[:, j] - theta[:, idx[na]]) / D2
                da2[:, j] = delta_star(p, xi, k2, D2)

    return db1, da1, db2, da2
