"""Named experiment scenarios for numerical studies.

Use this module to run the same notebook/script across multiple
parameter environments without hardcoding values inline.
"""

from copy import deepcopy

from asset.params import GAMMA, HY, IG, RHO, T


BASELINE = dict(
    IG=deepcopy(IG),
    HY=deepcopy(HY),
    GAMMA=float(GAMMA),
    RHO=float(RHO),
    T=float(T),
)


SCENARIOS = {
    "baseline": {},
    # Higher inventory aversion: tighter control, generally wider skew response.
    "high_risk_aversion": {
        "GAMMA": 1.8 * GAMMA,
    },
    # Lower inventory aversion: more aggressive risk-taking.
    "low_risk_aversion": {
        "GAMMA": 0.6 * GAMMA,
    },
    # More volatile market for both assets.
    "high_volatility": {
        "IG": {"sigma": 1.6 * IG["sigma"]},
        "HY": {"sigma": 1.6 * HY["sigma"]},
    },
    # Less liquid market: lower baseline flow and faster depth decay.
    "low_liquidity": {
        "IG": {"A": 0.65 * IG["A"], "k": 1.25 * IG["k"]},
        "HY": {"A": 0.65 * HY["A"], "k": 1.25 * HY["k"]},
    },
    # More liquid market: higher flow and slower depth decay.
    "high_liquidity": {
        "IG": {"A": 1.35 * IG["A"], "k": 0.85 * IG["k"]},
        "HY": {"A": 1.35 * HY["A"], "k": 0.85 * HY["k"]},
    },
    # Wider inventory bounds to test tail inventory behavior.
    "wide_inventory_limits": {
        "IG": {"Q": max(6, int(IG["Q"]))},
        "HY": {"Q": max(6, int(HY["Q"]))},
    },
    # Shorter horizon: stronger terminal effects.
    "short_horizon": {
        "T": 0.5 * T,
    },
    # Longer horizon: more accumulation risk and reversion dynamics.
    "long_horizon": {
        "T": 1.5 * T,
    },
    # Weak cross-asset linkage.
    "low_correlation": {
        "RHO": 0.2,
    },
    # Strong cross-asset linkage.
    "high_correlation": {
        "RHO": 0.98,
    },
}


def scenario_names():
    """Return available scenario names."""
    return sorted(SCENARIOS.keys())


def get_scenario(name):
    """Return merged parameter dict for a scenario name.

    Returns
    -------
    dict with keys: IG, HY, GAMMA, RHO, T, name
    """
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. Available: {scenario_names()}")

    out = dict(
        IG=deepcopy(BASELINE["IG"]),
        HY=deepcopy(BASELINE["HY"]),
        GAMMA=float(BASELINE["GAMMA"]),
        RHO=float(BASELINE["RHO"]),
        T=float(BASELINE["T"]),
        name=name,
    )
    patch = SCENARIOS[name]

    if "IG" in patch:
        out["IG"].update(patch["IG"])
    if "HY" in patch:
        out["HY"].update(patch["HY"])
    if "GAMMA" in patch:
        out["GAMMA"] = float(patch["GAMMA"])
    if "RHO" in patch:
        out["RHO"] = float(patch["RHO"])
    if "T" in patch:
        out["T"] = float(patch["T"])

    return out


def all_scenarios():
    """Return {name: merged_scenario_dict} for all scenarios."""
    return {name: get_scenario(name) for name in scenario_names()}
