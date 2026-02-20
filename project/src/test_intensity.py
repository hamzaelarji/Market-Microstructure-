"""Tests for intensity.py — verifies FOC and H values."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from intensity import Lambda, C_coeff, H_val, H_prime, H_second, delta_star
from params import IG, HY, GAMMA

def test_foc(xi, A, k, Delta, p_test, label=""):
    """Verify first‑order condition at δ*(p).
    
    The FOC of  sup_δ Λ(δ)/ξ · (1 − exp(−ξΔ(δ − p)))  is:
      Λ'(δ*)·[1 − exp(−ξΔ(δ*−p))] / ξ  +  Λ(δ*)·Δ·exp(−ξΔ(δ*−p))  = 0
    
    For ξ = 0 the FOC of  Δ·sup_δ Λ(δ)(δ − p)  is:
      Λ'(δ*)(δ*−p) + Λ(δ*) = 0
    """
    ds = delta_star(p_test, xi, k, Delta)
    xi_Delta = xi * Delta

    if abs(xi_Delta) < 1e-12:
        # ξ = 0 case
        foc = -k * Lambda(ds, A, k) * (ds - p_test) + Lambda(ds, A, k)
    else:
        e = np.exp(-xi_Delta * (ds - p_test))
        foc = (-k * Lambda(ds, A, k) / xi) * (1.0 - e) + Lambda(ds, A, k) * Delta * e

    # Also verify H value
    if abs(xi_Delta) < 1e-12:
        H_direct = Delta * Lambda(ds, A, k) * (ds - p_test)
    else:
        H_direct = (Lambda(ds, A, k) / xi) * (1.0 - np.exp(-xi_Delta * (ds - p_test)))

    H_formula = H_val(p_test, xi, A, k, Delta)
    rel_err_H = abs(H_direct - H_formula) / max(abs(H_formula), 1e-30)

    ok_foc = abs(foc) < 1e-9
    ok_H = rel_err_H < 1e-10
    status = "✓" if (ok_foc and ok_H) else "✗"

    print(f"  {status}  {label:30s}  |FOC|={abs(foc):.2e}   "
          f"|ΔH/H|={rel_err_H:.2e}   δ*={ds:.6e}   H={H_formula:.6e}")
    return ok_foc and ok_H


def test_limit_C():
    """C(ξΔ → 0) → e⁻¹."""
    for u in [1e-6, 1e-9, 1e-12, 0.0]:
        C = C_coeff(u, 1.79e4)
    ok = abs(C - np.exp(-1)) < 1e-10
    print(f"  {'✓' if ok else '✗'}  C(0) = {C:.10f}  vs  e⁻¹ = {np.exp(-1):.10f}")
    return ok


def test_derivatives(xi, A, k, Delta, p_test=0.0, label=""):
    """Verify H' and H'' by finite differences."""
    eps = 1e-8
    H0 = H_val(p_test, xi, A, k, Delta)
    Hp = H_val(p_test + eps, xi, A, k, Delta)
    Hm = H_val(p_test - eps, xi, A, k, Delta)

    fd_prime = (Hp - Hm) / (2 * eps)
    fd_second = (Hp - 2 * H0 + Hm) / eps ** 2

    an_prime = H_prime(p_test, xi, A, k, Delta)
    an_second = H_second(p_test, xi, A, k, Delta)

    err1 = abs(fd_prime - an_prime) / max(abs(an_prime), 1e-30)
    err2 = abs(fd_second - an_second) / max(abs(an_second), 1e-30)

    ok = err1 < 1e-5 and err2 < 1e-4
    print(f"  {'✓' if ok else '✗'}  {label:30s}  |ΔH'/H'|={err1:.2e}  |ΔH''/H''|={err2:.2e}")
    return ok


if __name__ == "__main__":
    all_ok = True
    print("=" * 80)
    print("TEST 1 — C coefficient limit")
    print("=" * 80)
    all_ok &= test_limit_C()

    print()
    print("=" * 80)
    print("TEST 2 — First‑Order Condition + H value")
    print("=" * 80)
    for p in [0.0, 1e-4, 5e-4, -2e-4]:
        # Model A — IG
        all_ok &= test_foc(GAMMA, IG['A'], IG['k'], IG['Delta'], p,
                           f"IG  Model A  p={p:.1e}")
        # Model A — HY
        all_ok &= test_foc(GAMMA, HY['A'], HY['k'], HY['Delta'], p,
                           f"HY  Model A  p={p:.1e}")
        # Model B — IG
        all_ok &= test_foc(0.0, IG['A'], IG['k'], IG['Delta'], p,
                           f"IG  Model B  p={p:.1e}")
        # Model B — HY
        all_ok &= test_foc(0.0, HY['A'], HY['k'], HY['Delta'], p,
                           f"HY  Model B  p={p:.1e}")

    print()
    print("=" * 80)
    print("TEST 3 — Derivatives (finite differences)")
    print("=" * 80)
    for p in [0.0, 1e-4]:
        all_ok &= test_derivatives(GAMMA, IG['A'], IG['k'], IG['Delta'], p,
                                   f"IG  Model A  p={p:.1e}")
        all_ok &= test_derivatives(0.0, HY['A'], HY['k'], HY['Delta'], p,
                                   f"HY  Model B  p={p:.1e}")

    print()
    print("=" * 80)
    print("TEST 4 — Numerical values (sanity)")
    print("=" * 80)
    xi = GAMMA
    for name, par in [("IG", IG), ("HY", HY)]:
        xi_Delta = xi * par['Delta']
        C = C_coeff(xi_Delta, par['k'])
        H0 = H_val(0, xi, par['A'], par['k'], par['Delta'])
        ds0 = delta_star(0, xi, par['k'], par['Delta'])
        print(f"  {name}:  ξΔ={xi_Delta:.1f}  C={C:.6f}  H(0)={H0:.6f}  δ*(0)={ds0:.3e}")

    print()
    if all_ok:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")