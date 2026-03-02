"""Paper trading bot using Binance websocket feed.

Usage (from project root):
    python -m bot.paper_trader --symbol BTCUSDT --gamma 1e-6 --T 3600
    python -m bot.paper_trader --simulated
"""

import asyncio
import json
import time
import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass, field

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ode_solver_1d import solve_general
from src.intensity import delta_star


@dataclass
class PaperTraderState:
    inventory: int = 0
    cash: float = 0.0
    mid_price: float = 0.0
    bid_quote: float = 0.0
    ask_quote: float = 0.0
    delta_bid: float = 0.0
    delta_ask: float = 0.0
    mtm: float = 0.0
    pnl: float = 0.0
    n_bid_fills: int = 0
    n_ask_fills: int = 0
    fees_paid: float = 0.0
    mtm_history: list = field(default_factory=list)
    inv_history: list = field(default_factory=list)
    price_history: list = field(default_factory=list)
    time_history: list = field(default_factory=list)
    is_running: bool = False
    status: str = "Initializing"
    start_time: float = 0.0
    last_update: float = 0.0


class PaperTrader:
    """Simulated market maker using optimal quotes on live data.

    1. Subscribes to Binance miniTicker for real-time mid prices
    2. Looks up optimal δ^b, δ^a from pre-solved ODE
    3. Simulates Poisson fills
    4. Tracks P&L, inventory, fills
    """

    def __init__(self, params, gamma=1e-6, T=3600.0, xi=None,
                 N_t=3600, maker_fee=-0.0001, symbol="BTCUSDT",
                 tick_interval=1.0):
        self.params = params
        self.gamma = gamma
        self.T = T
        self.xi = xi if xi is not None else gamma
        self.N_t = N_t
        self.maker_fee = maker_fee
        self.symbol = symbol.lower()
        self.tick_interval = tick_interval
        self.Delta = params["Delta"]
        self.Q = int(params["Q"])
        self.A = params["A"]
        self.k = params["k"]
        self.state = PaperTraderState()
        self._sol = None
        self._rng = np.random.default_rng(int(time.time()))

    def solve_ode(self):
        self.state.status = "Solving ODE..."
        self._sol = solve_general(self.params, self.gamma, self.T,
                                  xi=self.xi, N_t=self.N_t)
        self.state.status = "ODE solved, ready"

    def get_quotes(self, n, t_idx=0):
        i_lot = n + self.Q
        db = self._sol["delta_bid"][t_idx, i_lot]
        da = self._sol["delta_ask"][t_idx, i_lot]
        return (float(db) if np.isfinite(db) else np.inf,
                float(da) if np.isfinite(da) else np.inf)

    def process_tick(self, mid_price, dt):
        n = self.state.inventory
        db, da = self.get_quotes(n)
        self.state.mid_price = mid_price
        self.state.delta_bid = db
        self.state.delta_ask = da
        self.state.bid_quote = mid_price - db if db < np.inf else 0.0
        self.state.ask_quote = mid_price + da if da < np.inf else 0.0

        # Simulate fills
        if db < np.inf and n < self.Q:
            lam = self.A * np.exp(-self.k * max(db, 0.0))
            if self._rng.uniform() < 1.0 - np.exp(-lam * dt):
                fee = abs(self.maker_fee * (mid_price - db) * self.Delta)
                self.state.cash -= (mid_price - db) * self.Delta + fee
                self.state.inventory += 1
                self.state.n_bid_fills += 1
                self.state.fees_paid += fee

        if da < np.inf and n > -self.Q:
            lam = self.A * np.exp(-self.k * max(da, 0.0))
            if self._rng.uniform() < 1.0 - np.exp(-lam * dt):
                fee = abs(self.maker_fee * (mid_price + da) * self.Delta)
                self.state.cash += (mid_price + da) * self.Delta - fee
                self.state.inventory -= 1
                self.state.n_ask_fills += 1
                self.state.fees_paid += fee

        self.state.mtm = self.state.cash + self.state.inventory * self.Delta * mid_price
        self.state.pnl = self.state.mtm

        now = time.time()
        self.state.mtm_history.append(self.state.mtm)
        self.state.inv_history.append(self.state.inventory)
        self.state.price_history.append(mid_price)
        self.state.time_history.append(now - self.state.start_time)
        self.state.last_update = now

        max_hist = int(7200 / self.tick_interval)
        for attr in ("mtm_history", "inv_history", "price_history", "time_history"):
            lst = getattr(self.state, attr)
            if len(lst) > max_hist:
                setattr(self.state, attr, lst[-max_hist:])

    async def _run_websocket(self):
        import websockets
        url = f"wss://stream.binance.com:9443/ws/{self.symbol}@miniTicker"
        self.state.status = f"Connecting to {url}..."
        try:
            async with websockets.connect(url) as ws:
                self.state.status = f"Connected — trading {self.symbol.upper()}"
                self.state.is_running = True
                self.state.start_time = time.time()
                last_tick = time.time()
                async for msg in ws:
                    if not self.state.is_running:
                        break
                    data = json.loads(msg)
                    mid = (float(data["h"]) + float(data["l"])) / 2
                    now = time.time()
                    dt = now - last_tick
                    if dt >= self.tick_interval:
                        self.process_tick(mid, dt)
                        last_tick = now
        except Exception as e:
            self.state.status = f"Error: {e}"
            self.state.is_running = False

    def run_live(self):
        self.solve_ode()
        asyncio.run(self._run_websocket())

    def run_simulated(self, prices, dt=1.0):
        self.solve_ode()
        self.state.is_running = True
        self.state.start_time = time.time()
        self.state.status = "Running (simulated)"
        for p in prices:
            if not self.state.is_running:
                break
            self.process_tick(float(p), dt)
        self.state.is_running = False
        self.state.status = "Simulation complete"

    def stop(self):
        self.state.is_running = False
        self.state.status = "Stopped"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--gamma", type=float, default=1e-6)
    parser.add_argument("--T", type=float, default=3600.0)
    parser.add_argument("--simulated", action="store_true")
    args = parser.parse_args()

    from data.sample_data import CRYPTO_BTC

    trader = PaperTrader(params=CRYPTO_BTC, gamma=args.gamma,
                         T=args.T, symbol=args.symbol)
    if args.simulated:
        from data.sample_data import generate_trades
        from data.calibrate import compute_mid_price
        trades = generate_trades(T_hours=1)
        mid = compute_mid_price(trades, freq="1s")
        trader.run_simulated(mid.values, dt=1.0)
        s = trader.state
        print(f"P&L: {s.pnl:+.2f}  |  Fills: {s.n_bid_fills}b + {s.n_ask_fills}a  |  Inv: {s.inventory}")
    else:
        print(f"Starting live paper trading on {args.symbol}...")
        trader.run_live()
