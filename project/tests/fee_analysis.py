"""Fee analysis extension — impact of maker fees on strategy profitability.

Sweeps maker_fee from 0 to max_fee_bps and measures Sharpe, mean PnL, etc.
"""

import numpy as np
from market_making.simulation.backtest import BacktestConfig, run_backtest


def fee_sweep(params, gamma, fee_bps_range=None, T=3600.0,
              N_sim=300, seed=42, mid_prices=None):
    """Run backtest for each fee level and collect metrics.

    fee_bps_range: array of fees in basis points (e.g. [0, 0.5, 1, 2, 3, 5])

    Returns dict of arrays keyed by metric name.
    """
    if fee_bps_range is None:
        fee_bps_range = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])

    results = {
        "fee_bps": fee_bps_range,
        "mean_pnl": [],
        "std_pnl": [],
        "sharpe": [],
        "mean_fills": [],
        "gross_pnl": [],
        "total_fees": [],
        "net_pnl": [],
    }

    for fee_bps in fee_bps_range:
        maker_fee = fee_bps / 10_000.0

        cfg = BacktestConfig(
            params=params, gamma=gamma, T=T,
            N_sim=N_sim, seed=seed, strategy="optimal",
            maker_fee=maker_fee, mid_prices=mid_prices,
        )
        res = run_backtest(cfg)

        # Gross PnL (without fees) ≈ net PnL + fees paid
        net = res.pnl
        fees = res.fees_paid
        gross = net + fees

        results["mean_pnl"].append(float(np.mean(net)))
        results["std_pnl"].append(float(np.std(net)))
        s = float(np.std(net))
        results["sharpe"].append(float(np.mean(net) / s) if s > 0 else 0.0)
        results["mean_fills"].append(float(np.mean(res.n_bid_fills + res.n_ask_fills)))
        results["gross_pnl"].append(float(np.mean(gross)))
        results["total_fees"].append(float(np.mean(fees)))
        results["net_pnl"].append(float(np.mean(net)))

    for k in results:
        if k != "fee_bps":
            results[k] = np.array(results[k])

    return results


def find_breakeven_fee(params, gamma, T=3600.0, N_sim=300, seed=42,
                       max_bps=10.0, resolution=20):
    """Find the maker fee at which Sharpe ≈ 0 (break-even).

    Uses bisection-like sweep and interpolation.
    """
    fees = np.linspace(0, max_bps, resolution)
    res = fee_sweep(params, gamma, fee_bps_range=fees, T=T,
                    N_sim=N_sim, seed=seed)

    sharpes = res["sharpe"]
    # Find first crossing below 0
    for i in range(len(sharpes) - 1):
        if sharpes[i] >= 0 and sharpes[i + 1] < 0:
            # Linear interpolation
            f = sharpes[i] / max(sharpes[i] - sharpes[i + 1], 1e-12)
            return float(fees[i] + f * (fees[i + 1] - fees[i]))

    # Never crosses: either always profitable or always unprofitable
    return float(fees[-1]) if sharpes[0] >= 0 else 0.0
