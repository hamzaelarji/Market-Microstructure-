"""Generate synthetic trade data for demonstration.

Produces realistic-looking trade data with GBM prices + Poisson fills.
Use this when you don't have real Binance data downloaded yet.
"""

import numpy as np
import pandas as pd


def generate_trades(
    symbol: str = "BTCUSDT",
    S0: float = 95000.0,
    sigma_annual: float = 0.60,
    base_rate: float = 5.0,
    k_true: float = 0.05,
    T_hours: float = 24.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic aggTrades-like DataFrame."""
    rng = np.random.default_rng(seed)
    T_seconds = T_hours * 3600
    dt = 0.1

    sigma_ps = sigma_annual * S0 / np.sqrt(365.25 * 24 * 3600)
    n_steps = int(T_seconds / dt)
    dW = rng.standard_normal(n_steps) * np.sqrt(dt)

    prices = np.zeros(n_steps + 1)
    prices[0] = S0
    for i in range(n_steps):
        prices[i + 1] = prices[i] + sigma_ps * dW[i]

    max_rate = base_rate * 3
    records = []

    for i in range(n_steps):
        mid = prices[i]
        n_potential = rng.poisson(max_rate * dt)
        for _ in range(n_potential):
            delta = rng.exponential(1.0 / k_true)
            actual_rate = base_rate * np.exp(-k_true * delta)
            if rng.uniform() < actual_rate / max_rate:
                is_buy = rng.uniform() > 0.5
                trade_price = mid + delta if is_buy else mid - delta
                qty = rng.exponential(0.01)
                ts = pd.Timestamp("2025-01-15") + pd.Timedelta(seconds=i * dt)
                records.append({
                    "timestamp": ts,
                    "price": round(trade_price, 2),
                    "quantity": round(qty, 6),
                    "is_buyer_maker": not is_buy,
                })

    df = pd.DataFrame(records)
    return df.sort_values("timestamp").reset_index(drop=True)


def generate_pair(
    sym1="BTCUSDT", sym2="ETHUSDT",
    S0_1=95000.0, S0_2=3300.0,
    rho=0.85, T_hours=24.0, seed=42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate correlated pair of synthetic trade data."""
    rng = np.random.default_rng(seed)
    T_seconds = T_hours * 3600
    dt = 0.1

    sigma_ps_1 = 0.60 * S0_1 / np.sqrt(365.25 * 24 * 3600)
    sigma_ps_2 = 0.75 * S0_2 / np.sqrt(365.25 * 24 * 3600)
    n_steps = int(T_seconds / dt)

    Z1 = rng.standard_normal(n_steps)
    Z2 = rng.standard_normal(n_steps)
    W1 = Z1
    W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

    prices1 = np.zeros(n_steps + 1)
    prices2 = np.zeros(n_steps + 1)
    prices1[0], prices2[0] = S0_1, S0_2

    for i in range(n_steps):
        prices1[i + 1] = prices1[i] + sigma_ps_1 * np.sqrt(dt) * W1[i]
        prices2[i + 1] = prices2[i] + sigma_ps_2 * np.sqrt(dt) * W2[i]

    def _make_trades(prices, base_rate, k_true, rng_seed):
        _rng = np.random.default_rng(rng_seed)
        max_rate = base_rate * 3
        records = []
        for i in range(n_steps):
            mid = prices[i]
            for _ in range(_rng.poisson(max_rate * dt)):
                delta = _rng.exponential(1.0 / k_true)
                if _rng.uniform() < base_rate * np.exp(-k_true * delta) / max_rate:
                    is_buy = _rng.uniform() > 0.5
                    tp = mid + delta if is_buy else mid - delta
                    ts = pd.Timestamp("2025-01-15") + pd.Timedelta(seconds=i * dt)
                    records.append({"timestamp": ts, "price": round(tp, 2),
                                    "quantity": round(_rng.exponential(0.01), 6),
                                    "is_buyer_maker": not is_buy})
        return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)

    return (_make_trades(prices1, 5.0, 0.05, seed + 1),
            _make_trades(prices2, 8.0, 0.15, seed + 2))


# ── Pre-built crypto parameter sets (Guéant format) ─────────
CRYPTO_BTC = {
    "sigma": 10.12,       # $/√s
    "A": 5.0,             # 1/s
    "k": 0.05,            # 1/$
    "Delta": 9500.0,      # $
    "Q": 4,
}

CRYPTO_ETH = {
    "sigma": 0.44,        # $/√s
    "A": 8.0,             # 1/s
    "k": 0.15,            # 1/$
    "Delta": 330.0,       # $
    "Q": 4,
}

CRYPTO_GAMMA = 1e-6
CRYPTO_RHO = 0.85
CRYPTO_T = 3600
