"""Terminal penalty extension — explore ℓ(|q|) from Paper §2.

The terminal condition θ(T, n) = −ℓ(|nΔ|) changes quotes near T.
For large T, the penalty becomes irrelevant (asymptotic regime).
"""

import numpy as np
from market_making.core.solver_1d import solve_general


def ell_quadratic(c):
    """ℓ(|q|) = c · q².  Aggressive unwinding at T."""
    return lambda q_abs: c * q_abs ** 2


def ell_linear(c):
    """ℓ(|q|) = c · |q|.  Proportional penalty."""
    return lambda q_abs: c * q_abs


def solve_with_penalty(params, gamma, T, penalty_type="none", penalty_c=0.0,
                       xi=None, N_t=3600):
    """Solve ODE with specified terminal penalty.

    penalty_type: "none" | "linear" | "quadratic"
    penalty_c:    penalty coefficient
    """
    if xi is None:
        xi = gamma

    if penalty_type == "quadratic" and penalty_c > 0:
        ell = ell_quadratic(penalty_c)
    elif penalty_type == "linear" and penalty_c > 0:
        ell = ell_linear(penalty_c)
    else:
        ell = None

    return solve_general(params, gamma, T, xi=xi, N_t=N_t, ell_func=ell)


def penalty_convergence_sweep(params, gamma, xi=None,
                              T_values=None, penalty_c=1e-4):
    """Measure how fast penalty effect vanishes with increasing T.

    Returns dict with T_values, max_error (max|δ_penalty − δ_noPenalty| at t=0).
    """
    if xi is None:
        xi = gamma
    if T_values is None:
        T_values = np.array([600, 1200, 1800, 3600, 7200])

    errors = []
    for T in T_values:
        N_t = max(300, int(T))
        sol_no = solve_general(params, gamma, float(T), xi=xi, N_t=N_t)
        sol_yes = solve_with_penalty(params, gamma, float(T),
                                     penalty_type="quadratic", penalty_c=penalty_c,
                                     xi=xi, N_t=N_t)
        db_no = sol_no["delta_bid"][0, :]
        db_yes = sol_yes["delta_bid"][0, :]
        mask = np.isfinite(db_no) & np.isfinite(db_yes)
        if mask.any():
            errors.append(float(np.max(np.abs(db_no[mask] - db_yes[mask]))))
        else:
            errors.append(0.0)

    return {"T_values": T_values, "max_error": np.array(errors)}
