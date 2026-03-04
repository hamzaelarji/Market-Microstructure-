"""Optimal Market Making Dashboard.

Run:  streamlit run app.py
"""

import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.ode_solver_1d import solve_general
from src.closed_form import approx_quotes
from src.intensity import C_coeff
from data.calibrate import (
    calibrate, estimate_sigma, estimate_intensity, compute_mid_price,
)
from backtest.engine import BacktestConfig, run_backtest

# ─── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Market Making",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS: minimal polish ─────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.2rem; padding-bottom: 0.5rem;}
[data-testid="stMetric"] {
    background: rgba(128,128,128,0.06);
    border: 1px solid rgba(128,128,128,0.12);
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] {font-size: 0.75rem; opacity: 0.55;}
[data-testid="stMetricValue"] {font-size: 1.25rem;}
section[data-testid="stSidebar"] {display: none;}
hr {opacity: 0.12;}
</style>
""", unsafe_allow_html=True)


# ─── Load calibrated parameters ──────────────────────────
CALIBRATED_PATH = Path("data/data/calibrated/calibrated_params.json")
# st.write("cwd:", Path.cwd())
# st.write("CALIBRATED_PATH:", CALIBRATED_PATH)
# st.write("CALIBRATED_PATH resolved:", CALIBRATED_PATH.resolve())
# st.write("exists:", CALIBRATED_PATH.exists())

# st.write("exists:", CALIBRATED_PATH.exists())
# st.write("is_file:", CALIBRATED_PATH.is_file())
# st.write("parent exists:", CALIBRATED_PATH.parent.exists())

PALETTE = ["#00d4aa", "#ff6b6b", "#ffd93d"]


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


PARAMS, META, HAS_REAL_DATA = load_calibrated_params()
#debug
# st.write("HAS_REAL_DATA:", HAS_REAL_DATA)
# st.write("PARAMS keys:", list(PARAMS.keys()))
# st.write("META keys:", list(META.keys()))
# st.write("rho:", META.get("rho"))
SYMBOLS = list(PARAMS.keys())
DEFAULT_GAMMA = 0.01


# ─── Plotly defaults ─────────────────────────────────────
PLOT_KW = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=11),
    margin=dict(l=50, r=20, t=36, b=36),
)


def show(fig, **kw):
    """Render plotly with native theme disabled so our dark bg works."""
    st.plotly_chart(fig, use_container_width=True, theme="streamlit", **kw)


# ─── Header ──────────────────────────────────────────────
h1, h2 = st.columns([4, 1])
h1.markdown("### ◈ Optimal Market Making")
if HAS_REAL_DATA and SYMBOLS:
    m = META.get(SYMBOLS[0], {})
    h2.caption(f"Calibrated {m.get('calibration_date','—')} · {m.get('n_days','—')}d · "
               f"R²={m.get('r_squared','—')}")
else:
    h2.caption("⚠ Fallback params — run notebook 11")


# ─── Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Parameter Lab", "Calibration", "Backtest", "Paper Trading",
])


# ===========================================================
# TAB 1 — Parameter Lab
# ===========================================================
with tab1:
    c_ctrl, c_viz = st.columns([1, 3])

    with c_ctrl:
        preset = st.selectbox("Preset", SYMBOLS + ["Custom"])
        if preset in PARAMS:
            d = {**PARAMS[preset], "gamma": DEFAULT_GAMMA}
        else:
            d = {"sigma": 5.0, "A": 5.0, "k": 3.0,
                 "Delta": 100.0, "Q": 4, "gamma": 0.01}

        sigma = st.number_input("σ ($/√s)", value=d["sigma"], format="%.4f",
                                step=d["sigma"] * 0.1)
        A = st.number_input("A (1/s)", value=d["A"], format="%.4f",
                            step=d["A"] * 0.1)
        k = st.number_input("k (1/$)", value=d["k"], format="%.4f",
                            step=d["k"] * 0.1)
        Delta = st.number_input("Δ ($)", value=d["Delta"], format="%.2f",
                                step=d["Delta"] * 0.1)
        Q = st.slider("Q", 1, 10, d["Q"])
        gamma = st.number_input("γ", value=d["gamma"], format="%.4f", step=0.001)
        T_lab = st.number_input("T (s)", value=3600.0, step=600.0)
        model = st.radio("Model", ["A (ξ=γ)", "B (ξ=0)"], horizontal=True)
        xi = gamma if "A" in model else 0.0
        params = {"sigma": sigma, "A": A, "k": k, "Delta": Delta, "Q": Q}

    with c_viz:
        try:
            sol = solve_general(params, gamma, T_lab, xi=xi, N_t=3600)
            lots = sol["lots"]
            n_arr = np.arange(-Q + 1, Q)
            db_cf, da_cf = approx_quotes(n_arr, params, gamma, xi=xi)

            fig = make_subplots(rows=2, cols=2, vertical_spacing=0.12,
                horizontal_spacing=0.08,
                subplot_titles=["δ bid(0,n)", "δ ask(0,n)",
                                "Spread(0,n)", "Skew(0,n)"])

            db = sol["delta_bid"][0, :]
            da = sol["delta_ask"][0, :]

            def _add(row, col, x_ode, y_ode, y_cf, legend=False):
                m = np.isfinite(y_ode)
                fig.add_trace(go.Scatter(
                    x=x_ode[m], y=y_ode[m], mode="markers",
                    name="ODE", marker=dict(size=8, color=PALETTE[0]),
                    showlegend=legend), row=row, col=col)
                fig.add_trace(go.Scatter(
                    x=n_arr, y=y_cf, mode="lines",
                    name="Closed-form", line=dict(dash="dash", color=PALETTE[1]),
                    showlegend=legend), row=row, col=col)

            _add(1, 1, lots, db, db_cf, legend=True)
            _add(1, 2, lots, da, da_cf)
            sp = db + da
            _add(2, 1, lots, sp, db_cf + da_cf)
            sk = db - da
            _add(2, 2, lots, sk, db_cf - da_cf)

            fig.update_layout(**PLOT_KW, height=520,
                legend=dict(orientation="h", yanchor="bottom", y=1.02))
            for r in (1, 2):
                for c in (1, 2):
                    fig.update_xaxes(title_text="n", row=r, col=c)
            show(fig)

            # Derived metrics
            xi_D = xi * Delta
            d_s = (1.0 / xi_D) * np.log(1 + xi_D / k) if abs(xi_D) > 1e-12 else 1.0 / k
            C = C_coeff(xi_D, k)
            omega = np.sqrt(gamma * sigma**2 / (2.0 * A * Delta * k * C))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("δ_static", f"{d_s:.4f}")
            m2.metric("ω", f"{omega:.4e}")
            m3.metric("Spread(0)", f"{2 * d_s + omega * Delta:.4f}")
            m4.metric("ξΔ", f"{xi_D:.6f}")

        except Exception as e:
            st.error(f"ODE solve failed: {e}")


# ===========================================================
# TAB 2 — Calibration
# ===========================================================
with tab2:
    if HAS_REAL_DATA:
        rows = []
        for sym in SYMBOLS:
            p = PARAMS[sym]
            m = META.get(sym, {})
            rows.append({
                "Symbol": sym,
                "σ": f"{p['sigma']:.4f}",
                "A": f"{p['A']:.4f}",
                "k": f"{p['k']:.4f}",
                "Δ": f"{p['Delta']:.2f}",
                "R²": f"{m.get('r_squared', '—')}",
                "Trades": f"{m.get('n_trades', 0):,}",
                "Period": m.get("period", "—"),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Symbol"),
                     use_container_width=True)

        rho = META.get("rho")
        if rho:
            st.caption(f"ρ(BTC, ETH) = {rho:.4f}")

    st.divider()
    st.markdown("##### Re-calibrate from CSV")
    uploaded = st.file_uploader(
        "Upload trades CSV (timestamp, price, quantity)", type=["csv"])
    if uploaded:
        trades = pd.read_csv(uploaded)
        if not all(c in trades.columns for c in ["timestamp", "price", "quantity"]):
            st.error("CSV needs columns: timestamp, price, quantity")
        else:
            trades["timestamp"] = pd.to_datetime(trades["timestamp"])
            freq = st.selectbox("Sampling freq", ["1s", "5s", "10s"])
            if st.button("Calibrate", type="primary"):
                with st.spinner("Fitting..."):
                    cal = calibrate(trades, freq=freq)

                c1, c2, c3 = st.columns(3)
                c1.metric("σ", f"{cal.sigma:.4f}")
                c2.metric("A", f"{cal.A:.4f}")
                c3.metric("k", f"{cal.k:.4f}")

                st.session_state["calibrated_params"] = cal.to_dict()
                st.session_state["calibrated_params"]["Q"] = 4

                A_fit, k_fit, r_sq, bins, lam_obs = estimate_intensity(trades, freq)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=bins, y=lam_obs, name="Observed",
                                     opacity=0.5, marker_color=PALETTE[0]))
                xf = np.linspace(0, bins[-1], 200)
                fig.add_trace(go.Scatter(x=xf, y=A_fit * np.exp(-k_fit * xf),
                    mode="lines", name=f"Fit (R²={r_sq:.3f})",
                    line=dict(color=PALETTE[1], width=2)))
                fig.update_layout(**PLOT_KW, height=360,
                    xaxis_title="δ ($)", yaxis_title="Λ(δ)")
                show(fig)


# ===========================================================
# TAB 3 — Backtest
# ===========================================================
with tab3:
    c_set, c_res = st.columns([1, 3])

    with c_set:
        bt_sym = st.selectbox("Symbol", SYMBOLS, key="bt_sym")
        bt_params = st.session_state.get("calibrated_params", PARAMS[bt_sym]).copy()

        bt_gamma = st.number_input("γ", value=DEFAULT_GAMMA, format="%.4f", key="bt_g")
        bt_T = st.number_input("T (s)", value=3600.0, key="bt_t")
        bt_N = st.slider("Simulations", 100, 5000, 500, step=100)
        bt_fee = st.number_input("Maker fee (bps)", value=1.0, step=0.5) / 10000

        mid_real = load_mid_prices(bt_sym)
        use_real = False
        if mid_real is not None:
            use_real = st.checkbox("Use real prices", value=True)

        bt_strats = st.multiselect("Strategies",
            ["optimal", "naive", "closed_form"], default=["optimal", "naive"])

        run_bt = st.button("Run", type="primary", key="bt_run")

    with c_res:
        if run_bt:
            results = {}
            prog = st.progress(0)

            mid_prices_arr = None
            if use_real and mid_real is not None:
                N_t = int(bt_T)
                if len(mid_real) > N_t + 1:
                    idx0 = np.random.randint(0, len(mid_real) - N_t - 1)
                    mid_prices_arr = mid_real.values[idx0:idx0 + N_t + 1]

            for i, strat in enumerate(bt_strats):
                cfg = BacktestConfig(
                    params=bt_params, gamma=bt_gamma, T=bt_T,
                    N_sim=bt_N, maker_fee=bt_fee, strategy=strat, seed=42,
                    mid_prices=mid_prices_arr,
                )
                results[strat] = run_backtest(cfg)
                prog.progress((i + 1) / len(bt_strats))

            st.session_state["bt_results"] = results
            prog.empty()

        if "bt_results" in st.session_state:
            results = st.session_state["bt_results"]

            # Summary table
            rdf = pd.DataFrame([r.summary() for r in results.values()]).set_index("strategy")
            st.dataframe(rdf.style.format({
                "mean_pnl": "{:+,.2f}", "std_pnl": "{:,.2f}", "sharpe": "{:.3f}",
                "CE": "{:+,.2f}", "mean_fills": "{:.1f}", "mean_abs_inv_T": "{:.2f}",
                "max_drawdown": "{:,.2f}", "pct_flat": "{:.1%}", "mean_fees": "{:,.4f}",
            }), use_container_width=True)

            # P&L histogram
            fig = go.Figure()
            for i, (nm, r) in enumerate(results.items()):
                c = PALETTE[i % 3]
                fig.add_trace(go.Histogram(x=r.pnl, name=nm, opacity=0.5,
                    nbinsx=80, marker_color=c))
                fig.add_vline(x=r.mean_pnl, line_dash="dash", line_color=c,
                    annotation_text=f"μ={r.mean_pnl:+,.2f}")
            fig.update_layout(**PLOT_KW, height=320, xaxis_title="P&L ($)",
                              barmode="overlay")
            show(fig)

            # Inventory
            fig2 = make_subplots(rows=1, cols=2,
                subplot_titles=["E[|inventory|]", "Terminal inventory"])
            for i, (nm, r) in enumerate(results.items()):
                c = PALETTE[i % 3]
                step = max(1, r.inventory.shape[1] // 500)
                ma = np.mean(np.abs(r.inventory[:, ::step].astype(float)), axis=0)
                fig2.add_trace(go.Scatter(x=r.times[::step], y=ma, name=nm,
                    line=dict(color=c)), row=1, col=1)
                fig2.add_trace(go.Histogram(x=r.inventory[:, -1], name=nm,
                    opacity=0.5, showlegend=False, marker_color=c), row=1, col=2)
            fig2.update_layout(**PLOT_KW, height=320)
            show(fig2)

            # Sample trajectory
            traj = st.slider("Trajectory #", 0, bt_N - 1, bt_N // 2)
            fig3 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                subplot_titles=["Price", "Inventory", "MtM"],
                vertical_spacing=0.06)
            for i, (nm, r) in enumerate(results.items()):
                c = PALETTE[i % 3]
                fig3.add_trace(go.Scatter(x=r.times, y=r.price[traj],
                    showlegend=False, line=dict(color="#666", width=0.8)),
                    row=1, col=1)
                fig3.add_trace(go.Scatter(x=r.times, y=r.inventory[traj],
                    name=nm, line=dict(color=c, width=1.5)), row=2, col=1)
                fig3.add_trace(go.Scatter(x=r.times, y=r.mtm[traj],
                    showlegend=False, line=dict(color=c, width=1.5)),
                    row=3, col=1)
            fig3.update_layout(**PLOT_KW, height=550)
            show(fig3)


# ===========================================================
# TAB 4 — Paper Trading
# ===========================================================
with tab4:
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
            from bot.paper_trader import PaperTrader

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
