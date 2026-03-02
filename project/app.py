"""Streamlit dashboard for the Guéant Market Making Bot.

Run from project root:  streamlit run app.py

Tabs:
  1. Parameter Lab   — adjust σ, A, k, γ, see optimal quotes
  2. Calibration     — fit A, k, σ from trade data
  3. Backtest        — Monte Carlo, compare strategies
  4. Paper Trading   — simulated live trading
"""

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import time

from src.ode_solver_1d import solve_general
from src.closed_form import approx_quotes
from src.intensity import C_coeff

from data.sample_data import (
    generate_trades, generate_pair,
    CRYPTO_BTC, CRYPTO_ETH, CRYPTO_GAMMA, CRYPTO_T,
)
from data.calibrate import (
    calibrate, estimate_sigma, estimate_intensity, compute_mid_price,
)
from backtest.engine import BacktestConfig, run_backtest, compare_strategies


# ─── Page config ──────────────────────────────────────────
st.set_page_config(page_title="Guéant Market Making Bot", page_icon="📈", layout="wide")
st.title("📈 Optimal Market Making — Guéant (2017)")
st.caption("Interactive: parameter tuning · calibration · backtest · paper trading")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔧 Parameter Lab", "📊 Calibration", "🧪 Backtest", "🤖 Paper Trading",
])

# =============================================================
# TAB 1 — Parameter Lab
# =============================================================
with tab1:
    st.header("Parameter Lab")
    st.markdown("Adjust model parameters → ODE is re-solved live.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Model Parameters")
        preset = st.selectbox("Preset", ["Custom", "BTC/USDT", "ETH/USDT", "Paper IG", "Paper HY"])

        if preset == "BTC/USDT":
            defaults = {**CRYPTO_BTC, "gamma": CRYPTO_GAMMA}
        elif preset == "ETH/USDT":
            defaults = {**CRYPTO_ETH, "gamma": CRYPTO_GAMMA}
        elif preset == "Paper IG":
            defaults = {"sigma": 5.83e-6, "A": 9.10e-4, "k": 1.79e4,
                        "Delta": 5e7, "Q": 4, "gamma": 6e-5}
        elif preset == "Paper HY":
            defaults = {"sigma": 2.15e-5, "A": 3.40e-3, "k": 5.47e3,
                        "Delta": 1e7, "Q": 4, "gamma": 6e-5}
        else:
            defaults = {"sigma": 10.0, "A": 5.0, "k": 0.05,
                        "Delta": 9500.0, "Q": 4, "gamma": 1e-6}

        sigma = st.number_input("σ ($/√s)", value=defaults["sigma"],
                                format="%.6e", step=defaults["sigma"] * 0.1)
        A = st.number_input("A (1/s)", value=defaults["A"],
                            format="%.6e", step=defaults["A"] * 0.1)
        k = st.number_input("k (1/$)", value=defaults["k"],
                            format="%.6e", step=defaults["k"] * 0.1)
        Delta = st.number_input("Δ ($)", value=defaults["Delta"],
                                format="%.2f", step=defaults["Delta"] * 0.1)
        Q = st.slider("Q (max lots)", 1, 10, defaults["Q"])
        gamma = st.number_input("γ (1/$)", value=defaults.get("gamma", 1e-6),
                                format="%.6e", step=defaults.get("gamma", 1e-6) * 0.1)
        T_lab = st.number_input("T (s)", value=3600.0, step=600.0)
        model = st.radio("Model", ["Model A (ξ=γ)", "Model B (ξ=0)"], horizontal=True)
        xi = gamma if "A" in model else 0.0
        params = {"sigma": sigma, "A": A, "k": k, "Delta": Delta, "Q": Q}

    with col2:
        st.subheader("Optimal Quotes")
        with st.spinner("Solving ODE..."):
            try:
                sol = solve_general(params, gamma, T_lab, xi=xi, N_t=3600)
                lots = sol["lots"]
                n_arr = np.arange(-Q + 1, Q)
                db_cf, da_cf = approx_quotes(n_arr, params, gamma, xi=xi)

                fig = make_subplots(rows=2, cols=2,
                    subplot_titles=["δ^bid(0,n)", "δ^ask(0,n)", "Spread(0,n)", "Skew(0,n)"])

                db = sol["delta_bid"][0, :]
                da = sol["delta_ask"][0, :]

                # Bid
                m = np.isfinite(db)
                fig.add_trace(go.Scatter(x=lots[m], y=db[m], mode="markers",
                    name="Exact (ODE)", marker=dict(symbol="x", size=10)), row=1, col=1)
                fig.add_trace(go.Scatter(x=n_arr, y=db_cf, mode="lines",
                    name="Closed-form", line=dict(dash="dash")), row=1, col=1)
                # Ask
                m = np.isfinite(da)
                fig.add_trace(go.Scatter(x=lots[m], y=da[m], mode="markers",
                    name="Exact", marker=dict(symbol="circle", size=10),
                    showlegend=False), row=1, col=2)
                fig.add_trace(go.Scatter(x=n_arr, y=da_cf, mode="lines",
                    name="CF", line=dict(dash="dash"), showlegend=False), row=1, col=2)
                # Spread
                sp = db + da; m = np.isfinite(sp)
                fig.add_trace(go.Scatter(x=lots[m], y=sp[m], mode="markers",
                    marker=dict(symbol="square", size=8), showlegend=False), row=2, col=1)
                fig.add_trace(go.Scatter(x=n_arr, y=db_cf+da_cf, mode="lines",
                    line=dict(dash="dash"), showlegend=False), row=2, col=1)
                # Skew
                sk = db - da; m = np.isfinite(sk)
                fig.add_trace(go.Scatter(x=lots[m], y=sk[m], mode="markers",
                    marker=dict(symbol="diamond", size=8), showlegend=False), row=2, col=2)
                fig.add_trace(go.Scatter(x=n_arr, y=db_cf-da_cf, mode="lines",
                    line=dict(dash="dash"), showlegend=False), row=2, col=2)

                fig.update_layout(height=600, template="plotly_white")
                for i in range(1, 3):
                    for j in range(1, 3):
                        fig.update_xaxes(title_text="n (lots)", row=i, col=j)
                st.plotly_chart(fig, use_container_width=True)

                # Derived quantities
                xi_Delta = xi * Delta
                d_static = (1.0 / xi_Delta) * np.log(1.0 + xi_Delta / k) if abs(xi_Delta) > 1e-12 else 1.0 / k
                C = C_coeff(xi_Delta, k)
                omega = np.sqrt(gamma * sigma**2 / (2.0 * A * Delta * k * C))

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("δ_static", f"{d_static:.6e}")
                c2.metric("ω (slope)", f"{omega:.6e}")
                c3.metric("Spread(0)", f"{2*d_static + omega*Delta:.6e}")
                c4.metric("ξΔ", f"{xi_Delta:.4f}")
            except Exception as e:
                st.error(f"ODE solve failed: {e}")


# =============================================================
# TAB 2 — Calibration
# =============================================================
with tab2:
    st.header("Calibration from Trade Data")
    st.markdown("Estimate (σ, A, k) from real or synthetic trades.")

    data_source = st.radio("Source", ["Synthetic (demo)", "Upload CSV"], horizontal=True)

    if data_source == "Synthetic (demo)":
        c1, c2 = st.columns(2)
        with c1:
            sym = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT"])
            T_hours = st.slider("Hours of data", 1, 48, 24)
        with c2:
            S0 = st.number_input("Price ($)", value=95000.0 if sym == "BTCUSDT" else 3300.0)
            vol_a = st.slider("Annual vol (%)", 20, 150, 60) / 100

        if st.button("Generate & Calibrate", type="primary"):
            with st.spinner("Generating..."):
                trades = generate_trades(symbol=sym, S0=S0, sigma_annual=vol_a, T_hours=T_hours)
            st.success(f"{len(trades):,} trades generated")

            with st.spinner("Calibrating..."):
                cal = calibrate(trades, freq="1s", symbol=sym)

            c1, c2, c3 = st.columns(3)
            c1.metric("σ ($/√s)", f"{cal.sigma:.4e}")
            c2.metric("A (1/s)", f"{cal.A:.4e}")
            c3.metric("k (1/$)", f"{cal.k:.4e}")

            st.session_state["calibrated_params"] = cal.to_dict()
            st.session_state["calibrated_params"]["Q"] = 4

            # Intensity fit plot
            st.subheader("Intensity Fit: Λ(δ) = A·exp(-k·δ)")
            A_fit, k_fit, r_sq, bins, lam_obs = estimate_intensity(trades, "1s")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=bins, y=lam_obs, name="Observed", opacity=0.6))
            x_f = np.linspace(0, bins[-1], 200)
            fig.add_trace(go.Scatter(x=x_f, y=A_fit * np.exp(-k_fit * x_f),
                mode="lines", name=f"Fit (R²={r_sq:.3f})", line=dict(color="red", width=2)))
            fig.update_layout(xaxis_title="δ ($)", yaxis_title="Intensity (1/s)",
                              template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Mid-price
            mid = compute_mid_price(trades, "1s")
            fig2 = make_subplots(rows=1, cols=2, subplot_titles=["Mid-price", "Returns"])
            fig2.add_trace(go.Scatter(y=mid.values, mode="lines"), row=1, col=1)
            fig2.add_trace(go.Histogram(x=mid.diff().dropna().values, nbinsx=100), row=1, col=2)
            fig2.update_layout(height=350, template="plotly_white", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        uploaded = st.file_uploader("Upload trades CSV", type=["csv"])
        if uploaded:
            trades = pd.read_csv(uploaded)
            if not all(c in trades.columns for c in ["timestamp", "price", "quantity"]):
                st.error("CSV needs: timestamp, price, quantity")
            else:
                trades["timestamp"] = pd.to_datetime(trades["timestamp"])
                freq = st.selectbox("Freq", ["1s", "5s", "10s"])
                if st.button("Calibrate"):
                    cal = calibrate(trades, freq=freq)
                    st.json(cal.to_dict())
                    st.session_state["calibrated_params"] = cal.to_dict()


# =============================================================
# TAB 3 — Backtest
# =============================================================
with tab3:
    st.header("Monte Carlo Backtest")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Settings")
        bt_params = st.session_state.get("calibrated_params", CRYPTO_BTC).copy()
        if "calibrated_params" in st.session_state:
            st.info("Using calibrated params")

        bt_gamma = st.number_input("γ", value=CRYPTO_GAMMA, format="%.2e", key="bt_g")
        bt_T = st.number_input("T (s)", value=float(CRYPTO_T), key="bt_t")
        bt_N = st.slider("N_sim", 100, 5000, 500, step=100)
        bt_fee = st.number_input("Maker fee (bps)", value=1.0, step=0.5) / 10000
        bt_strats = st.multiselect("Strategies",
            ["optimal", "naive", "closed_form"], default=["optimal", "naive"])
        run_bt = st.button("Run Backtest", type="primary")

    with c2:
        if run_bt:
            results = {}
            prog = st.progress(0)
            for i, strat in enumerate(bt_strats):
                with st.spinner(f"Running {strat}..."):
                    cfg = BacktestConfig(params=bt_params, gamma=bt_gamma, T=bt_T,
                                         N_sim=bt_N, maker_fee=bt_fee, strategy=strat, seed=42)
                    results[strat] = run_backtest(cfg)
                prog.progress((i + 1) / len(bt_strats))
            st.session_state["bt_results"] = results

        if "bt_results" in st.session_state:
            results = st.session_state["bt_results"]

            st.subheader("Performance")
            rows = [r.summary() for r in results.values()]
            df = pd.DataFrame(rows).set_index("strategy")
            st.dataframe(df.style.format({
                "mean_pnl": "{:+,.0f}", "std_pnl": "{:,.0f}", "sharpe": "{:.3f}",
                "CE": "{:+,.0f}", "mean_fills": "{:.1f}", "mean_abs_inv_T": "{:.2f}",
                "max_drawdown": "{:,.0f}", "pct_flat": "{:.1%}", "mean_fees": "{:,.2f}",
            }), use_container_width=True)

            # P&L histogram
            fig = go.Figure()
            for nm, r in results.items():
                fig.add_trace(go.Histogram(x=r.pnl, name=nm, opacity=0.6, nbinsx=80))
                fig.add_vline(x=r.mean_pnl, line_dash="dash",
                    annotation_text=f"{nm} μ={r.mean_pnl:+,.0f}")
            fig.update_layout(xaxis_title="P&L ($)", barmode="overlay",
                              template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Inventory control
            fig2 = make_subplots(rows=1, cols=2,
                subplot_titles=["E[|inventory|]", "Terminal inventory"])
            for nm, r in results.items():
                step = max(1, r.inventory.shape[1] // 500)
                ma = np.mean(np.abs(r.inventory[:, ::step].astype(float)), axis=0)
                fig2.add_trace(go.Scatter(x=r.times[::step], y=ma, name=nm), row=1, col=1)
                fig2.add_trace(go.Histogram(x=r.inventory[:, -1], name=nm,
                    opacity=0.5, showlegend=False), row=1, col=2)
            fig2.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)

            # Sample trajectory
            st.subheader("Sample Trajectory")
            traj = st.slider("Trajectory #", 0, bt_N - 1, bt_N // 2)
            fig3 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                subplot_titles=["Price", "Inventory", "MtM"])
            for nm, r in results.items():
                fig3.add_trace(go.Scatter(x=r.times, y=r.price[traj],
                    name=f"price", showlegend=False), row=1, col=1)
                fig3.add_trace(go.Scatter(x=r.times, y=r.inventory[traj], name=nm), row=2, col=1)
                fig3.add_trace(go.Scatter(x=r.times, y=r.mtm[traj], name=nm,
                    showlegend=False), row=3, col=1)
            fig3.update_layout(height=700, template="plotly_white")
            st.plotly_chart(fig3, use_container_width=True)


# =============================================================
# TAB 4 — Paper Trading
# =============================================================
with tab4:
    st.header("Paper Trading Simulator")

    c1, c2 = st.columns([1, 2])
    with c1:
        pt_sym = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT"], key="pt_s")
        pt_gamma = st.number_input("γ", value=1e-6, format="%.2e", key="pt_g")
        pt_hours = st.slider("Duration (h)", 1, 24, 4, key="pt_h")
        pt_params = st.session_state.get("calibrated_params",
            CRYPTO_BTC if pt_sym == "BTCUSDT" else CRYPTO_ETH).copy()
        run_pt = st.button("Start Simulation", type="primary")

    with c2:
        if run_pt:
            from bot.paper_trader import PaperTrader

            with st.spinner("Generating prices..."):
                S0 = 95000.0 if pt_sym == "BTCUSDT" else 3300.0
                trades = generate_trades(symbol=pt_sym, S0=S0, T_hours=pt_hours)
                mid = compute_mid_price(trades, freq="1s").dropna()

            with st.spinner("Solving ODE & running..."):
                trader = PaperTrader(params=pt_params, gamma=pt_gamma,
                                     T=pt_hours * 3600, symbol=pt_sym)
                trader.run_simulated(mid.values, dt=1.0)

            s = trader.state
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("P&L", f"${s.pnl:+,.2f}")
            c2.metric("Inventory", f"{s.inventory} lots")
            c3.metric("Bid Fills", s.n_bid_fills)
            c4.metric("Ask Fills", s.n_ask_fills)

            t = np.array(s.time_history)
            if len(t) > 0:
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=["Price ($)", "Inventory (lots)", "MtM ($)"])
                fig.add_trace(go.Scatter(x=t/3600, y=s.price_history,
                    mode="lines", line=dict(width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=t/3600, y=s.inv_history,
                    mode="lines", line=dict(width=1.5)), row=2, col=1)
                fig.add_trace(go.Scatter(x=t/3600, y=s.mtm_history,
                    mode="lines", line=dict(width=1.5, color="green")), row=3, col=1)
                fig.update_xaxes(title_text="Time (hours)", row=3, col=1)
                fig.update_layout(height=700, template="plotly_white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("""
    ### 🔌 Live Trading
    ```bash
    python -m bot.paper_trader --symbol BTCUSDT --gamma 1e-6 --T 3600
    ```
    Connects to Binance public websocket (no API key). No real orders placed.
    """)


# ─── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    ### About
    [Guéant (2017)](https://arxiv.org/abs/1510.01767) implementation:
    ODE solver · closed-form · calibration · backtest · paper trading

    **Data:** [Binance Public](https://data.binance.vision/) or synthetic

    ---
    """)
    if "calibrated_params" in st.session_state:
        st.markdown("### Active Params")
        st.json(st.session_state["calibrated_params"])
