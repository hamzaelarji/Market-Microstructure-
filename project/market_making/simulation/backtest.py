"""Backtest engine for Guéant optimal market making.

Supports: Poisson fills, maker fees, multiple strategies (optimal, naive, closed-form).
Convention: inventory in lots, n ∈ {-Q, ..., +Q}.
"""

import numpy as np
from dataclasses import dataclass, field

from market_making.core.solver_1d import solve_general
from market_making.core.closed_form import approx_quotes
from market_making.core.intensity import C_coeff, fill_prob


@dataclass
class BacktestConfig:
    params: dict                      # {sigma, A, k, Delta, Q}
    gamma: float = 1e-6
    xi: float = None                  # None → gamma (Model A)
    T: float = 3600.0
    N_t: int = 3600
    N_sim: int = 1000
    maker_fee: float = 0.0001        # 1 bps
    seed: int = 42
    strategy: str = "optimal"         # optimal | naive | closed_form
    mid_prices: np.ndarray = None     # if provided, replay real prices


@dataclass
class BacktestResult:
    pnl: np.ndarray
    inventory: np.ndarray
    cash: np.ndarray
    mtm: np.ndarray
    price: np.ndarray
    times: np.ndarray
    n_bid_fills: np.ndarray
    n_ask_fills: np.ndarray
    fees_paid: np.ndarray
    strategy: str = ""
    config: BacktestConfig = None

    @property
    def mean_pnl(self): return float(np.mean(self.pnl))
    @property
    def std_pnl(self): return float(np.std(self.pnl))
    @property
    def sharpe(self):
        s = self.std_pnl
        return self.mean_pnl / s if s > 0 else 0.0
    @property
    def max_drawdown(self):
        dd = np.zeros(len(self.pnl))
        for i in range(len(self.pnl)):
            peak = np.maximum.accumulate(self.mtm[i])
            dd[i] = np.max(peak - self.mtm[i])
        return float(np.mean(dd))
    @property
    def mean_fills(self):
        return float(np.mean(self.n_bid_fills + self.n_ask_fills))
    @property
    def mean_abs_inventory(self):
        return float(np.mean(np.abs(self.inventory[:, -1])))

    def certainty_equivalent(self, gamma=None):
        if gamma is None:
            gamma = self.config.gamma if self.config else 1e-6
        max_val = np.max(-gamma * self.pnl)
        log_E = max_val + np.log(np.mean(np.exp(-gamma * self.pnl - max_val)))
        return -log_E / gamma

    def summary(self) -> dict:
        return {
            "strategy": self.strategy,
            "mean_pnl": self.mean_pnl,
            "std_pnl": self.std_pnl,
            "sharpe": self.sharpe,
            "CE": self.certainty_equivalent(),
            "mean_fills": self.mean_fills,
            "mean_abs_inv_T": self.mean_abs_inventory,
            "max_drawdown": self.max_drawdown,
            "pct_flat": float(np.mean(self.inventory[:, -1] == 0)),
            "mean_fees": float(np.mean(self.fees_paid)),
        }


def run_backtest(config: BacktestConfig) -> BacktestResult:
    """Run a full Monte Carlo backtest."""
    params = config.params
    gamma = config.gamma
    xi = config.xi if config.xi is not None else gamma
    T, N_t, N_sim = config.T, config.N_t, config.N_sim
    sigma, A, k = params["sigma"], params["A"], params["k"]
    Delta, Q = params["Delta"], int(params["Q"])
    dt = T / N_t
    rng = np.random.default_rng(config.seed)

    # ── Compute quote tables ──
    if config.strategy in ("optimal", "naive"):
        sol = solve_general(params, gamma, T, xi=xi, N_t=N_t)
        db_table = sol["delta_bid"]
        da_table = sol["delta_ask"]

    if config.strategy == "naive":
        half_spread = db_table[0, Q]

    if config.strategy == "closed_form":
        n_arr = np.arange(-Q, Q + 1)
        db_cf, da_cf = approx_quotes(n_arr, params, gamma, xi=xi)

    # ── Real prices (optional) ──
    base_price = None
    if config.mid_prices is not None:
        real = config.mid_prices
        if len(real) != N_t + 1:
            xp = np.linspace(0, 1, len(real))
            x = np.linspace(0, 1, N_t + 1)
            base_price = np.interp(x, xp, real)
        else:
            base_price = real.copy()

    # ── Allocate arrays ──
    price = np.zeros((N_sim, N_t + 1))
    cash = np.zeros((N_sim, N_t + 1))
    inv = np.zeros((N_sim, N_t + 1), dtype=int)
    mtm = np.zeros((N_sim, N_t + 1))
    n_bid_fills = np.zeros(N_sim, dtype=int)
    n_ask_fills = np.zeros(N_sim, dtype=int)
    fees_paid = np.zeros(N_sim)

    dW    = rng.standard_normal((N_sim, N_t))
    U_bid = rng.uniform(size=(N_sim, N_t))
    U_ask = rng.uniform(size=(N_sim, N_t))

    for m in range(N_sim):
        S = 0.0 if base_price is None else base_price[0]
        X, n = 0.0, 0
        price[m, 0] = S

        for t_idx in range(N_t):
            i_lot = n + Q

            # Get quotes
            if config.strategy == "optimal":
                db = db_table[t_idx, i_lot] if (n < Q and np.isfinite(db_table[t_idx, i_lot])) else np.inf
                da = da_table[t_idx, i_lot] if (n > -Q and np.isfinite(da_table[t_idx, i_lot])) else np.inf
            elif config.strategy == "naive":
                db = half_spread if n < Q else np.inf
                da = half_spread if n > -Q else np.inf
            else:
                db = db_cf[i_lot] if n < Q else np.inf
                da = da_cf[i_lot] if n > -Q else np.inf

            # Price update
            if base_price is not None:
                S = base_price[t_idx + 1]
            else:
                S += sigma * np.sqrt(dt) * dW[m, t_idx]

            # Fills — using fill_prob() from core/intensity (single source of truth)
            if db < np.inf:
                lam = A * np.exp(-k * max(db, 0.0))
                if U_bid[m, t_idx] < fill_prob(lam, dt):
                    fee = config.maker_fee * abs(S - db) * Delta
                    X -= (S - db) * Delta + fee
                    n += 1
                    n_bid_fills[m] += 1
                    fees_paid[m] += fee

            if da < np.inf:
                lam = A * np.exp(-k * max(da, 0.0))
                if U_ask[m, t_idx] < fill_prob(lam, dt):
                    fee = config.maker_fee * abs(S + da) * Delta
                    X += (S + da) * Delta - fee
                    n -= 1
                    n_ask_fills[m] += 1
                    fees_paid[m] += fee

            price[m, t_idx + 1] = S
            cash[m, t_idx + 1] = X
            inv[m, t_idx + 1] = n
            mtm[m, t_idx + 1] = X + n * Delta * S

    times = np.linspace(0, T, N_t + 1)
    pnl = cash[:, -1] + inv[:, -1] * Delta * price[:, -1]

    return BacktestResult(
        pnl=pnl, inventory=inv, cash=cash, mtm=mtm, price=price,
        times=times, n_bid_fills=n_bid_fills, n_ask_fills=n_ask_fills,
        fees_paid=fees_paid, strategy=config.strategy, config=config,
    )


def compare_strategies(
    params: dict, gamma=1e-6, T=3600.0, N_sim=1000, seed=42,
    mid_prices=None, strategies=None,
) -> dict[str, BacktestResult]:
    """Run multiple strategies and return results keyed by strategy name."""
    if strategies is None:
        strategies = ["optimal", "naive", "closed_form"]
    results = {}
    for strat in strategies:
        cfg = BacktestConfig(
            params=params, gamma=gamma, T=T,
            N_sim=N_sim, seed=seed, strategy=strat,
            mid_prices=mid_prices,
        )
        results[strat] = run_backtest(cfg)
    return results
