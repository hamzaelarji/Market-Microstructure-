"""ui/tabs/paper_trading.py — Tab 8: Simulated live paper trading."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_making.data.calibrate import compute_mid_price
from market_making.bot.paper_trader import PaperTrader
from ui.styles import hero_banner

DEFAULT_GAMMA = 0.01


def render_paper_trading(PARAMS, META, HAS_REAL_DATA, PALETTE, PLOT_KW, show,
                         load_mid_prices, run_quick_mc):
    """Render Tab 8 — Paper Trading: simulated live trading session."""
    hero_banner(
        "📈",
        "Paper Trading",
        "Simulated live market-making session using the optimal quoting policy.",
    )
    SYMBOLS = list(PARAMS.keys())

    c_pt, c_out = st.columns([1, 3])

    with c_pt:
        pt_sym = st.selectbox("Symbol", SYMBOLS, key="pt_s")
        pt_gamma = st.number_input("γ", value=DEFAULT_GAMMA, format="%.4f", key="pt_g")
        pt_hours = st.slider("Duration (h)", 1, 24, 4, key="pt_h")
        pt_params = st.session_state.get("calibrated_params", PARAMS[pt_sym]).copy()

        mid_real = load_mid_prices(pt_sym)
        pt_use_real = False
        if mid_real is not None:
            pt_use_real = st.checkbox("Use real prices", value=True, key="pt_real")

        run_pt = st.button("Start", type="primary", key="pt_run")

    with c_out:
        if run_pt:
            with st.spinner("Loading prices..."):
                if pt_use_real and mid_real is not None:
                    N_pts = pt_hours * 3600
                    if len(mid_real) > N_pts:
                        idx0 = np.random.randint(0, len(mid_real) - N_pts)
                        mid_vals = mid_real.values[idx0:idx0 + N_pts]
                    else:
                        mid_vals = mid_real.values[:N_pts]
                else:
                    from data.sample_data import generate_trades
                    mp = META.get(pt_sym, {}).get("mean_price", 95000)
                    trades = generate_trades(symbol=pt_sym, S0=mp,
                                             T_hours=pt_hours)
                    mid_vals = compute_mid_price(trades, "1s").dropna().values

            with st.spinner("Solving & running..."):
                trader = PaperTrader(params=pt_params, gamma=pt_gamma,
                                     T=pt_hours * 3600, symbol=pt_sym)
                trader.run_simulated(mid_vals, dt=1.0)

            s = trader.state
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("P&L", f"${s.pnl:+,.2f}")
            m2.metric("Inventory", f"{s.inventory}")
            m3.metric("Bid Fills", s.n_bid_fills)
            m4.metric("Ask Fills", s.n_ask_fills)

            t = np.array(s.time_history)
            if len(t) > 0:
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=["Price (USDT)", "Inventory", "MtM ($)"],
                    vertical_spacing=0.06)
                fig.add_trace(go.Scatter(x=t / 3600, y=s.price_history,
                    line=dict(width=0.8, color="#666")), row=1, col=1)
                fig.add_trace(go.Scatter(x=t / 3600, y=s.inv_history,
                    line=dict(width=1.5, color=PALETTE[0])), row=2, col=1)
                fig.add_trace(go.Scatter(x=t / 3600, y=s.mtm_history,
                    line=dict(width=1.5, color=PALETTE[2])), row=3, col=1)
                fig.update_xaxes(title_text="Hours", row=3, col=1)
                fig.update_layout(**PLOT_KW, height=550, showlegend=False)
                show(fig)

        st.divider()
        st.caption("Live mode: `python -m bot.paper_trader --symbol BTCUSDT`")
