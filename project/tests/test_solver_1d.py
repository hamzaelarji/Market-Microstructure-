"""Consistency and smoke tests for 1D/2D ODE solvers."""

import numpy as np

from market_making.params.assets import GAMMA, HY, IG
from market_making.core.solver_1d import solve_general, solve_model_a, solve_model_b
from market_making.core.solver_2d import solve_2d


def _max_abs_diff(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.nan
    return float(np.max(np.abs(x[mask] - y[mask])))


def test_1d_model_a_consistency():
    """Model A: specialised linear solver should match general Newton solver."""
    T_test = 600.0
    N_t = 240

    sol_a = solve_model_a(IG, GAMMA, T_test, N_t=N_t)
    sol_g = solve_general(IG, GAMMA, T_test, xi=GAMMA, N_t=N_t)

    # Theta can differ by a state-dependent level; quotes are the target objects.
    theta_a_centered = sol_a["theta"] - sol_a["theta"][:, [0]]
    theta_g_centered = sol_g["theta"] - sol_g["theta"][:, [0]]
    err_theta = _max_abs_diff(theta_a_centered, theta_g_centered)
    err_bid = _max_abs_diff(sol_a["delta_bid"], sol_g["delta_bid"])
    err_ask = _max_abs_diff(sol_a["delta_ask"], sol_g["delta_ask"])

    ok = (err_bid < 5e-6) and (err_ask < 5e-6)
    print(
        f"  {'✓' if ok else '✗'}  1D Model A consistency  "
        f"|Δθ_centered|max={err_theta:.2e}  |Δδb|max={err_bid:.2e}  |Δδa|max={err_ask:.2e}"
    )
    return ok


def test_1d_model_b_consistency():
    """Model B: specialised Newton solver should match general solver with xi=0."""
    T_test = 600.0
    N_t = 240

    sol_b = solve_model_b(HY, GAMMA, T_test, N_t=N_t)
    sol_g = solve_general(HY, GAMMA, T_test, xi=0.0, N_t=N_t)

    err_theta = _max_abs_diff(sol_b["theta"], sol_g["theta"])
    err_bid = _max_abs_diff(sol_b["delta_bid"], sol_g["delta_bid"])
    err_ask = _max_abs_diff(sol_b["delta_ask"], sol_g["delta_ask"])

    ok = (err_theta < 1e-8) and (err_bid < 1e-8) and (err_ask < 1e-8)
    print(
        f"  {'✓' if ok else '✗'}  1D Model B consistency  "
        f"|Δθ|max={err_theta:.2e}  |Δδb|max={err_bid:.2e}  |Δδa|max={err_ask:.2e}"
    )
    return ok


def test_1d_quote_boundaries():
    """Boundary states should have one-sided quotes only."""
    sol = solve_general(IG, GAMMA, 300.0, xi=GAMMA, N_t=120)

    bid_right_nan = np.all(np.isnan(sol["delta_bid"][:, -1]))  # n=+Q cannot bid more
    ask_left_nan = np.all(np.isnan(sol["delta_ask"][:, 0]))    # n=-Q cannot ask more
    bid_interior_finite = np.all(np.isfinite(sol["delta_bid"][0, :-1]))
    ask_interior_finite = np.all(np.isfinite(sol["delta_ask"][0, 1:]))

    ok = bid_right_nan and ask_left_nan and bid_interior_finite and ask_interior_finite
    print(
        f"  {'✓' if ok else '✗'}  1D boundary quotes  "
        f"bid@+Q nan={bid_right_nan}  ask@-Q nan={ask_left_nan}"
    )
    return ok


def test_2d_solver_smoke():
    """Fast structural test for 2D solver outputs and boundaries."""
    p1 = {**IG, "Q": 1}
    p2 = {**HY, "Q": 1}
    sol = solve_2d(p1, p2, GAMMA, rho=0.3, T=300.0, xi=GAMMA, N_t=40)

    n_grid = (2 * p1["Q"] + 1) * (2 * p2["Q"] + 1)
    expected_shape = (41, n_grid)

    shape_ok = (
        sol["theta"].shape == expected_shape
        and sol["delta_bid_1"].shape == expected_shape
        and sol["delta_ask_1"].shape == expected_shape
        and sol["delta_bid_2"].shape == expected_shape
        and sol["delta_ask_2"].shape == expected_shape
    )
    finite_ok = (
        np.isfinite(sol["delta_bid_1"][0]).any()
        and np.isfinite(sol["delta_ask_1"][0]).any()
        and np.isfinite(sol["delta_bid_2"][0]).any()
        and np.isfinite(sol["delta_ask_2"][0]).any()
    )

    q1 = p1["Q"]
    q2 = p2["Q"]
    bid1_boundary_nan = all(
        np.isnan(sol["delta_bid_1"][:, j]).all()
        for (n1, n2), j in sol["idx"].items()
        if n1 == q1
    )
    ask1_boundary_nan = all(
        np.isnan(sol["delta_ask_1"][:, j]).all()
        for (n1, n2), j in sol["idx"].items()
        if n1 == -q1
    )
    bid2_boundary_nan = all(
        np.isnan(sol["delta_bid_2"][:, j]).all()
        for (n1, n2), j in sol["idx"].items()
        if n2 == q2
    )
    ask2_boundary_nan = all(
        np.isnan(sol["delta_ask_2"][:, j]).all()
        for (n1, n2), j in sol["idx"].items()
        if n2 == -q2
    )

    ok = (
        shape_ok
        and finite_ok
        and bid1_boundary_nan
        and ask1_boundary_nan
        and bid2_boundary_nan
        and ask2_boundary_nan
    )
    print(
        f"  {'✓' if ok else '✗'}  2D smoke test  "
        f"shape_ok={shape_ok}  finite_ok={finite_ok}"
    )
    return ok


if __name__ == "__main__":
    all_ok = True
    print("=" * 80)
    print("ODE SOLVER TESTS")
    print("=" * 80)
    all_ok &= test_1d_model_a_consistency()
    all_ok &= test_1d_model_b_consistency()
    all_ok &= test_1d_quote_boundaries()
    all_ok &= test_2d_solver_smoke()

    print()
    if all_ok:
        print("ALL ODE SOLVER TESTS PASSED ✓")
    else:
        print("SOME ODE SOLVER TESTS FAILED ✗")
