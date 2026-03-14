"""Calibrate Guéant model parameters (σ, A, k) from trade data.

    σ : from mid-price return volatility
    A, k : by fitting Λ(δ) = A·exp(-k·δ) to empirical fill rates
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from dataclasses import dataclass, asdict


@dataclass
class CalibratedParams:
    sigma: float
    A: float
    k: float
    Delta: float
    Q: int
    symbol: str = ""
    dt_seconds: float = 1.0
    n_trades: int = 0
    r_squared: float = 0.0

    def to_dict(self):
        return {k: v for k, v in asdict(self).items()
                if k in ("sigma", "A", "k", "Delta", "Q")}


def compute_mid_price(trades: pd.DataFrame, freq: str = "1s") -> pd.Series:
    """VWAP per interval as mid-price proxy."""
    trades = trades.set_index("timestamp")
    notional = (trades["price"] * trades["quantity"]).resample(freq).sum()
    volume = trades["quantity"].resample(freq).sum()
    return (notional / volume).dropna()

def compute_mid_price_tick(df: pd.DataFrame) -> pd.Series:
    """
    Estimate mid price tick-by-tick using buyer/seller alternation.
    Buyer-maker = sell aggressor → price is near ask.
    Seller-maker = buy aggressor → price is near bid.
    Mid ≈ rolling mean of consecutive bid/ask prices.
    """
    df = df.sort_values("timestamp").copy()
    # Simple proxy: mid = rolling 2-trade mean price
    df["mid"] = df["price"].rolling(2, min_periods=1).mean()
    return df.set_index("timestamp")["mid"]


def estimate_sigma(trades: pd.DataFrame, freq: str = "1s") -> float:
    """σ = std(ΔS) / √dt in $/√s."""
    mid = compute_mid_price(trades, freq)
    returns = mid.diff().dropna()
    if freq.endswith("s"):
        dt = float(freq[:-1]) if len(freq) > 1 else 1.0
    elif freq.endswith("min") or freq.endswith("T"):
        dt = 60.0
    else:
        dt = 1.0
    return float(returns.std() / np.sqrt(dt))


def estimate_intensity(
    trades: pd.DataFrame,
    freq: str = "1s",
    n_bins: int = 20,
    max_delta_quantile: float = 0.95,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Fit Λ(δ) = A·exp(-k·δ) to trade data.

    Returns (A, k, R², bin_centers, lambda_observed).
    """
    mid = compute_mid_price(trades, freq)
    trades_a = trades.set_index("timestamp").copy()
    trades_a["mid"] = mid.reindex(trades_a.index, method="ffill")
    trades_a = trades_a.dropna(subset=["mid"])
    trades_a["delta"] = np.abs(trades_a["price"] - trades_a["mid"])

    deltas = trades_a["delta"].values
    max_delta = np.quantile(deltas[deltas > 0], max_delta_quantile)
    bins = np.linspace(0, max_delta, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    total_seconds = (trades["timestamp"].max() - trades["timestamp"].min()).total_seconds()
    counts = np.histogram(deltas, bins=bins)[0]
    lambda_obs = counts / total_seconds

    mask = lambda_obs > 0
    if mask.sum() < 3:
        raise ValueError("Not enough non-empty bins for fitting")

    x_fit, y_fit = bin_centers[mask], lambda_obs[mask]

    def exp_model(d, A, k):
        return A * np.exp(-k * d)

    try:
        A0 = y_fit[0]
        k0 = 1.0 / (max_delta * 0.3) if max_delta > 0 else 1.0
        popt, _ = curve_fit(exp_model, x_fit, y_fit, p0=[A0, k0],
                            bounds=([0, 0], [np.inf, np.inf]), maxfev=5000)
        A_fit, k_fit = popt
        y_pred = exp_model(x_fit, A_fit, k_fit)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except RuntimeError:
        log_y = np.log(y_fit)
        coeffs = np.polyfit(x_fit, log_y, 1)
        k_fit, A_fit, r_sq = -coeffs[0], np.exp(coeffs[1]), 0.0

    return float(A_fit), float(k_fit), float(r_sq), bin_centers, lambda_obs


def calibrate(
    trades: pd.DataFrame,
    Delta: float = None,
    Q: int = 4,
    freq: str = "1s",
    symbol: str = "",
) -> CalibratedParams:
    """Full calibration: estimate σ, A, k from trade data."""
    sigma = estimate_sigma(trades, freq)
    A, k, r_sq, _, _ = estimate_intensity(trades, freq)

    if Delta is None:
        Delta = round(trades["price"].mean() * 0.1, 2)

    return CalibratedParams(
        sigma=sigma, A=A, k=k, Delta=Delta, Q=Q,
        symbol=symbol, n_trades=len(trades), r_squared=r_sq,
    )


def calibrate_pair(
    trades1: pd.DataFrame, trades2: pd.DataFrame,
    freq="1s", sym1="ASSET1", sym2="ASSET2",
    Delta1=None, Delta2=None, Q=4,
) -> tuple[CalibratedParams, CalibratedParams, float]:
    """Calibrate two assets + their return correlation."""
    p1 = calibrate(trades1, Delta=Delta1, Q=Q, freq=freq, symbol=sym1)
    p2 = calibrate(trades2, Delta=Delta2, Q=Q, freq=freq, symbol=sym2)

    mid1 = compute_mid_price(trades1, freq)
    mid2 = compute_mid_price(trades2, freq)
    common = mid1.index.intersection(mid2.index)
    if len(common) < 10:
        rho = 0.0
    else:
        r1 = mid1.reindex(common).diff().dropna()
        r2 = mid2.reindex(common).diff().dropna()
        c = r1.index.intersection(r2.index)
        rho = float(np.corrcoef(r1.loc[c], r2.loc[c])[0, 1])

    return p1, p2, rho


def print_params(params: CalibratedParams, gamma: float = 1e-6):
    print(f"  Symbol:  {params.symbol}")
    print(f"  σ  = {params.sigma:.6e}  ($/√s)")
    print(f"  A  = {params.A:.6e}  (1/s)")
    print(f"  k  = {params.k:.6e}  (1/$)")
    print(f"  Δ  = {params.Delta:.2f}      ($)")
    print(f"  Q  = {params.Q}")
    print(f"  R² = {params.r_squared:.4f}")
    print(f"  n  = {params.n_trades:,} trades")
    print(f"  ξΔ = {gamma * params.Delta:.4f}  (at γ={gamma:.1e})")
