"""ui/loaders.py — Cached data loaders and quick-MC helper."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from market_making.simulation.backtest import BacktestConfig, run_backtest

CALIBRATED_PATH = Path("data/data/calibrated/calibrated_params.json")


@st.cache_data
def load_calibrated_params():
    """Load from notebook 11, or fall back to defaults."""
    if CALIBRATED_PATH.exists():
        with open(CALIBRATED_PATH) as f:
            raw = json.load(f)
        params, meta = {}, {}
        for key, val in raw.items():
            if key == "cross_correlation":
                meta["rho"] = val.get("rho", 0.85)
            elif isinstance(val, dict) and "sigma" in val:
                params[key] = {
                    "sigma": val["sigma"], "A": val["A"], "k": val["k"],
                    "Delta": val["Delta"], "Q": val.get("Q", 4),
                }
                meta[key] = {
                    k2: val[k2] for k2 in
                    ["mean_price", "r_squared", "n_trades", "n_days",
                     "total_hours", "calibration_date", "period", "lot_size"]
                    if k2 in val
                }
        return params, meta, True
    else:
        params = {
            "BTCUSDT": {"sigma": 5.76, "A": 5.55, "k": 2.73, "Delta": 91.86, "Q": 4},
            "ETHUSDT": {"sigma": 0.24, "A": 1.65, "k": 11.62, "Delta": 20.23, "Q": 4},
        }
        return params, {}, False


@st.cache_data
def load_mid_prices(symbol):
    p = Path(f"data/data/calibrated/mid_prices_{symbol}.parquet")
    if p.exists():
        return pd.read_parquet(p)["mid_price"]
    return None


def run_quick_mc(params, gamma, T=3600.0, N_sim=500, seed=42, mid_prices=None):
    """Quick MC for a single γ. Returns summary dict with all key metrics."""
    cfg = BacktestConfig(
        params=params, gamma=gamma, T=T,
        N_sim=N_sim, seed=seed, strategy="optimal",
        mid_prices=mid_prices,
    )
    res = run_backtest(cfg)
    pnl = res.pnl
    fills = res.n_bid_fills + res.n_ask_fills
    inv_final = res.inventory[:, -1]

    var5 = float(np.percentile(pnl, 5))
    dd_list = []
    for i in range(N_sim):
        peak = np.maximum.accumulate(res.mtm[i])
        dd_list.append(np.max(peak - res.mtm[i]))

    ce = res.certainty_equivalent(gamma)

    return {
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": float(np.std(pnl)),
        "sharpe": float(np.mean(pnl) / np.std(pnl)) if np.std(pnl) > 0 else 0.0,
        "var5": var5,
        "max_dd": float(np.mean(dd_list)),
        "ce": ce,
        "mean_fills": float(np.mean(fills)),
        "mean_abs_inv": float(np.mean(np.abs(inv_final))),
        "fill_rate_per_h": float(np.mean(fills)) / max(T / 3600, 1e-6),
    }
