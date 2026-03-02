"""Tests for intensity_hawkes.py.

Verifies:
  1. When excitation = 0, lambda_hawkes ≈ baseline  (Softplus(Λ) ≈ Λ for Λ≫1;
     and that the function is monotone / correct shape).
  2. Excitation decay dynamics match the analytical exponential kernel.
  3. Jump updates (self / cross) are additive and correct.
  4. fill_prob_from_intensity is exact Poisson (not first-order approx).
  5. HawkesState step() integrates decay + jumps in the right order.
  6. Linear vs Softplus comparison — Softplus is always ≥ 0.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intensity_hawkes import (
    softplus,
    softplus_deriv,
    lambda_hawkes,
    lambda_hawkes_linear,
    decay_excitation,
    update_excitation_bid_fill,
    update_excitation_ask_fill,
    fill_prob_from_intensity,
    HawkesState,
    DEFAULT_HAWKES_CFG,
)

# Simple exponential baseline (replicates intensity.py without importing it)
def _Lambda(delta, A=1.0, k=1.0):
    return A * np.exp(-k * delta)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def _check(condition: bool, label: str) -> bool:
    status = "✓" if condition else "✗"
    print(f"  {status}  {label}")
    return condition


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1 — Softplus properties
# ═════════════════════════════════════════════════════════════════════════════
def test_softplus_properties() -> bool:
    print("=" * 72)
    print("TEST 1 — Softplus properties")
    print("=" * 72)
    ok = True

    # 1a. softplus(0) = ln(2)
    val = softplus(0.0)
    ok &= _check(abs(val - np.log(2)) < 1e-12,
                 f"softplus(0) = ln(2) = {np.log(2):.6f}  got {val:.6f}")

    # 1b. softplus(x) ≈ x for large x
    for x in [20.0, 50.0, 100.0]:
        v = softplus(x)
        ok &= _check(abs(v - x) < 1e-6, f"softplus({x}) ≈ {x}  got {v:.6f}")

    # 1c. softplus is always positive
    for x in [-100.0, -10.0, 0.0, 10.0, 100.0]:
        ok &= _check(softplus(x) > 0, f"softplus({x}) > 0  got {softplus(x):.2e}")

    # 1d. softplus_deriv(x) ∈ (0,1)
    for x in [-5.0, 0.0, 5.0]:
        d = softplus_deriv(x)
        ok &= _check(0.0 < d < 1.0, f"softplus'({x}) ∈ (0,1)  got {d:.6f}")

    # 1e. numerical derivative check
    eps = 1e-7
    for x in [-3.0, 0.0, 3.0]:
        fd = (softplus(x + eps) - softplus(x - eps)) / (2 * eps)
        an = softplus_deriv(x)
        err = abs(fd - an) / max(abs(an), 1e-12)
        ok &= _check(err < 1e-5, f"softplus' FD check at x={x}  err={err:.2e}")

    return ok


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2 — Zero-excitation: lambda_hawkes reduces to baseline shape
# ═════════════════════════════════════════════════════════════════════════════
def test_zero_excitation() -> bool:
    print()
    print("=" * 72)
    print("TEST 2 — Zero excitation: λ_hawkes(Λ, 0) = softplus(Λ)")
    print("=" * 72)
    ok = True

    A, k = 140.0, 1.5
    for delta in [0.0, 0.5, 1.0, 2.0]:
        baseline = _Lambda(delta, A, k)
        lh = lambda_hawkes(baseline, 0.0)
        sp = softplus(baseline)
        err = abs(lh - sp)
        ok &= _check(err < 1e-14,
                     f"δ={delta:.1f}  Λ={baseline:.4f}  λ_H={lh:.6f}  "
                     f"softplus(Λ)={sp:.6f}  err={err:.2e}")

    # When Λ is large, softplus(Λ) ≈ Λ
    for delta in [0.0, 0.1]:
        baseline = _Lambda(delta, A=1e4, k=1.0)
        lh = lambda_hawkes(baseline, 0.0)
        rel = abs(lh - baseline) / baseline
        ok &= _check(rel < 1e-6,
                     f"Large Λ={baseline:.2e}: λ_H≈Λ  rel_err={rel:.2e}")

    return ok


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3 — Excitation decay dynamics
# ═════════════════════════════════════════════════════════════════════════════
def test_excitation_decay() -> bool:
    print()
    print("=" * 72)
    print("TEST 3 — Excitation decay dynamics")
    print("=" * 72)
    ok = True

    beta = 5.0
    y0   = 3.0
    dt   = 0.01

    # After n steps, y(n·dt) = y0 · exp(−β · n · dt)
    y = y0
    for n in range(1, 6):
        y = decay_excitation(y, beta, dt)
        expected = y0 * np.exp(-beta * n * dt)
        err = abs(y - expected)
        ok &= _check(err < 1e-12,
                     f"After {n} step(s):  y={y:.8f}  expected={expected:.8f}  "
                     f"err={err:.2e}")

    # Decay to near-zero over long time
    y_long = y0 * np.exp(-beta * 10.0)
    ok &= _check(y_long < 1e-20, f"Long-time decay y(10s)={y_long:.2e} ≈ 0")

    return ok


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4 — Jump updates (self / cross excitation)
# ═════════════════════════════════════════════════════════════════════════════
def test_jump_updates() -> bool:
    print()
    print("=" * 72)
    print("TEST 4 — Jump updates (self / cross excitation)")
    print("=" * 72)
    ok = True

    alpha_s = 2.0
    alpha_c = 0.5
    y_bid, y_ask = 1.0, 0.5

    # Bid fill
    yb2, ya2 = update_excitation_bid_fill(y_bid, y_ask, alpha_s, alpha_c)
    ok &= _check(abs(yb2 - (y_bid + alpha_s)) < 1e-14,
                 f"Bid fill: y_bid {y_bid} → {yb2}  (expected {y_bid+alpha_s})")
    ok &= _check(abs(ya2 - (y_ask + alpha_c)) < 1e-14,
                 f"Bid fill: y_ask {y_ask} → {ya2}  (expected {y_ask+alpha_c})")

    # Ask fill — roles reversed
    yb3, ya3 = update_excitation_ask_fill(y_bid, y_ask, alpha_s, alpha_c)
    ok &= _check(abs(yb3 - (y_bid + alpha_c)) < 1e-14,
                 f"Ask fill: y_bid {y_bid} → {yb3}  (expected {y_bid+alpha_c})")
    ok &= _check(abs(ya3 - (y_ask + alpha_s)) < 1e-14,
                 f"Ask fill: y_ask {y_ask} → {ya3}  (expected {y_ask+alpha_s})")

    # Symmetry: bid then ask vs ask then bid should give same total excitation
    # (since each adds α_self + α_cross to the same pool, ignoring order)
    tot_bid_ask = (y_bid + alpha_s + alpha_c)
    tot_ask_bid = (y_bid + alpha_c + alpha_s)
    ok &= _check(abs(tot_bid_ask - tot_ask_bid) < 1e-14,
                 "Total excitation commutative (no-interaction case)")

    return ok


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5 — Fill probability (exact Poisson)
# ═════════════════════════════════════════════════════════════════════════════
def test_fill_probability() -> bool:
    print()
    print("=" * 72)
    print("TEST 5 — Fill probability: P(fill) = 1 − exp(−λ·dt)")
    print("=" * 72)
    ok = True

    dt = 0.1
    for lam in [0.01, 0.1, 1.0, 10.0, 100.0]:
        p = fill_prob_from_intensity(lam, dt)
        p_exact = 1.0 - np.exp(-lam * dt)
        err = abs(p - p_exact)
        ok &= _check(err < 1e-14,
                     f"λ={lam:6.2f}  P={p:.8f}  exact={p_exact:.8f}  err={err:.2e}")

    # P ∈ [0, 1]
    for lam in [0.0, 1e-6, 1e6]:
        p = fill_prob_from_intensity(lam, dt)
        ok &= _check(0.0 <= p <= 1.0, f"λ={lam:.2e}  P={p:.6f} ∈ [0,1]")

    # First-order approx error at large λ
    lam_large = 50.0
    p_exact_large   = fill_prob_from_intensity(lam_large, dt)
    p_first_order   = min(lam_large * dt, 1.0)
    err_fo = abs(p_exact_large - p_first_order)
    ok &= _check(err_fo > 5e-3,
                 f"Exact vs first-order differ at λ={lam_large}: "
                 f"Δ={err_fo:.4f} > 5e-3  (confirms exact formula needed)")

    return ok


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6 — HawkesState integration
# ═════════════════════════════════════════════════════════════════════════════
def test_hawkes_state() -> bool:
    print()
    print("=" * 72)
    print("TEST 6 — HawkesState integration")
    print("=" * 72)
    ok = True

    cfg = {"beta": 5.0, "alpha_self": 2.0, "alpha_cross": 0.5}
    state = HawkesState(cfg)
    mu = 1.0
    dt = 0.01

    # 6a. No fills → excitation stays zero, intensity = softplus(mu)
    state.reset()
    for _ in range(10):
        state.step(dt, fill_bid=False, fill_ask=False)
    ok &= _check(state.y_bid == 0.0 and state.y_ask == 0.0,
                 "No fills → y_bid = y_ask = 0")
    lam_b = state.lambda_bid(mu)
    ok &= _check(abs(lam_b - softplus(mu)) < 1e-14,
                 f"No excitation: λ_bid = softplus({mu}) = {softplus(mu):.6f}  "
                 f"got {lam_b:.6f}")

    # 6b. After one bid fill, y_bid increases, then decays
    state.reset()
    state.step(dt, fill_bid=True, fill_ask=False)
    # After decay then jump:
    y_bid_expected = 0.0 * np.exp(-cfg["beta"] * dt) + cfg["alpha_self"]
    y_ask_expected = 0.0 * np.exp(-cfg["beta"] * dt) + cfg["alpha_cross"]
    ok &= _check(abs(state.y_bid - y_bid_expected) < 1e-12,
                 f"After 1 bid fill: y_bid={state.y_bid:.6f}  "
                 f"expected={y_bid_expected:.6f}")
    ok &= _check(abs(state.y_ask - y_ask_expected) < 1e-12,
                 f"After 1 bid fill: y_ask={state.y_ask:.6f}  "
                 f"expected={y_ask_expected:.6f}")

    # 6c. Intensity > baseline after excitation
    lam_after = state.lambda_bid(mu)
    lam_base  = softplus(mu)
    ok &= _check(lam_after > lam_base,
                 f"Post-fill: λ_bid={lam_after:.4f} > baseline={lam_base:.4f}")

    # 6d. Long-run decay: intensity returns to baseline
    state.reset()
    state.step(dt, fill_bid=True, fill_ask=False)
    for _ in range(10_000):
        state.step(dt, fill_bid=False, fill_ask=False)
    lam_long = state.lambda_bid(mu)
    ok &= _check(abs(lam_long - softplus(mu)) < 1e-6,
                 f"Long-run: λ_bid → softplus({mu})={softplus(mu):.6f}  "
                 f"got {lam_long:.8f}")

    # 6e. backward-compat aliases (jump_self / jump_cross)
    cfg_legacy = {"beta": 5.0, "jump_self": 1.0, "jump_cross": 0.3}
    s2 = HawkesState(cfg_legacy)
    ok &= _check(s2.alpha_self == 1.0 and s2.alpha_cross == 0.3,
                 f"Legacy alias: alpha_self={s2.alpha_self}  "
                 f"alpha_cross={s2.alpha_cross}")

    return ok


# ═════════════════════════════════════════════════════════════════════════════
# TEST 7 — Softplus vs linear: Softplus always ≥ 0
# ═════════════════════════════════════════════════════════════════════════════
def test_nonnegativity() -> bool:
    print()
    print("=" * 72)
    print("TEST 7 — Non-negativity: Softplus vs linear Hawkes")
    print("=" * 72)
    ok = True

    # Large negative excitation can make linear go negative
    baseline    = 0.1
    excitation  = -10.0
    lam_sp  = lambda_hawkes(baseline, excitation)
    lam_lin = lambda_hawkes_linear(baseline, excitation)

    ok &= _check(lam_sp > 0,
                 f"Softplus: λ({baseline}, {excitation}) = {lam_sp:.6e} > 0")
    ok &= _check(lam_lin == 0.0,
                 f"Linear clipped to 0: λ({baseline}, {excitation}) = {lam_lin}")
    ok &= _check(lam_sp > lam_lin,
                 "Softplus > linear for negative-dominated input (inhibition regime)")

    # For positive excitation, both should be close
    excitation_pos = 5.0
    lh  = lambda_hawkes(baseline, excitation_pos)
    lhl = lambda_hawkes_linear(baseline, excitation_pos)
    ok &= _check(abs(lh - (baseline + excitation_pos)) < 0.01,
                 f"Large positive: Softplus≈linear={baseline+excitation_pos:.2f}  "
                 f"got {lh:.6f}")

    return ok


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = [
        test_softplus_properties(),
        test_zero_excitation(),
        test_excitation_decay(),
        test_jump_updates(),
        test_fill_probability(),
        test_hawkes_state(),
        test_nonnegativity(),
    ]

    print()
    print("=" * 72)
    if all(results):
        print("ALL HAWKES INTENSITY TESTS PASSED ✓")
    else:
        n_fail = results.count(False)
        print(f"{n_fail} TEST(S) FAILED ✗")
