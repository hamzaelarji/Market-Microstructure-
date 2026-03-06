"""Optimal Market Making Dashboard.

Run:  streamlit run app.py

Tabs:
  1. Parameter Lab      — ODE quotes, 3D surface, Model A vs B
  2. γ Sensitivity       — sweep γ → spread, skew, fill rate
  3. Efficient Frontier  — MC E[P&L] vs Std[P&L] for 12 γ values
  4. Calibration         — show calibrated params / re-calibrate
  5. Backtest            — MC backtest, strategy comparison
  6. Intraday Regimes    — static vs regime-aware policy
  7. Hawkes vs Poisson   — compare fill dynamics
  8. Paper Trading       — simulated live trading
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
from data.calibrate import (calibrate, estimate_sigma, estimate_intensity, compute_mid_price)
from backtest.engine import BacktestConfig, run_backtest

try:
    from src.intensity_hawkes import (
        HawkesState, lambda_hawkes, softplus,
        fill_prob_from_intensity, DEFAULT_HAWKES_CFG,
    )
    HAWKES_AVAILABLE = True
except ImportError:
    HAWKES_AVAILABLE = False


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


# ─── Helpers for new tabs ────────────────────────────────

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Parameter Lab",
    "γ Sensitivity",
    "Efficient Frontier",
    "Calibration",
    "Backtest",
    "Intraday Regimes",
    "Hawkes vs Poisson",
    "Paper Trading",
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

        show_both = st.checkbox("Compare A vs B side-by-side", value=False)
        show_3d = st.checkbox("Show 3D δ(t,n) surface", value=False)

    with c_viz:
        try:
            N_t_lab = max(300, int(T_lab))
            sol = solve_general(params, gamma, T_lab, xi=xi, N_t=N_t_lab)
            lots = sol["lots"]
            n_arr = np.arange(-Q + 1, Q)
            db_cf, da_cf = approx_quotes(n_arr, params, gamma, xi=xi)

            db = sol["delta_bid"][0, :]
            da = sol["delta_ask"][0, :]

            if show_both:
                xi_b = 0.0 if "A" in model else gamma
                sol_b = solve_general(params, gamma, T_lab, xi=xi_b, N_t=N_t_lab)
                db_b = sol_b["delta_bid"][0, :]
                da_b = sol_b["delta_ask"][0, :]

            fig = make_subplots(rows=2, cols=2, vertical_spacing=0.12,
                horizontal_spacing=0.08,
                subplot_titles=["δ bid(0,n)", "δ ask(0,n)",
                                "Spread(0,n)", "Skew(0,n)"])

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

            def _add_b(row, col, x_ode, y_ode):
                m = np.isfinite(y_ode)
                fig.add_trace(go.Scatter(
                    x=x_ode[m], y=y_ode[m], mode="markers",
                    name="Other model", marker=dict(size=6, color=PALETTE[2],
                    symbol="circle-open"),
                    showlegend=(row == 1 and col == 1)), row=row, col=col)

            _add(1, 1, lots, db, db_cf, legend=True)
            _add(1, 2, lots, da, da_cf)
            _add(2, 1, lots, db + da, db_cf + da_cf)
            _add(2, 2, lots, db - da, db_cf - da_cf)

            if show_both:
                _add_b(1, 1, lots, db_b)
                _add_b(1, 2, lots, da_b)
                _add_b(2, 1, lots, db_b + da_b)
                _add_b(2, 2, lots, db_b - da_b)

            fig.update_layout(**PLOT_KW, height=520,
                legend=dict(orientation="h", yanchor="bottom", y=1.02))
            for r in (1, 2):
                for c in (1, 2):
                    fig.update_xaxes(title_text="n", row=r, col=c)
            show(fig)

            xi_D = xi * Delta
            d_s = (1.0 / xi_D) * np.log(1 + xi_D / k) if abs(xi_D) > 1e-12 else 1.0 / k
            C = C_coeff(xi_D, k)
            omega = np.sqrt(gamma * sigma**2 / (2.0 * A * Delta * k * C))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("δ_static", f"{d_s:.4f}")
            m2.metric("ω", f"{omega:.4e}")
            m3.metric("Spread(0)", f"{2 * d_s + omega * Delta:.4f}")
            m4.metric("ξΔ", f"{xi_D:.6f}")

            if show_3d:
                times_3d = sol["times"]
                db_3d = sol["delta_bid"]
                step_t = max(1, len(times_3d) // 80)
                t_sub = times_3d[::step_t]
                db_sub = db_3d[::step_t, :]
                db_plot = np.where(np.isfinite(db_sub), db_sub, np.nan)

                fig3d = go.Figure(data=[go.Surface(
                    z=db_plot, x=lots, y=t_sub,
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(title="δ^bid"),
                )])
                fig3d.update_layout(
                    **PLOT_KW, height=500,
                    scene=dict(
                        xaxis_title="n (lots)",
                        yaxis_title="t (s)",
                        zaxis_title="δ^bid",
                        bgcolor="rgba(0,0,0,0)",
                    ),
                )
                show(fig3d)

        except Exception as e:
            st.error(f"ODE solve failed: {e}")


# ===========================================================
# TAB 2 — γ Sensitivity  (G1)
# ===========================================================
with tab2:
    st.caption(
        "How does risk aversion γ control the market maker? "
        "Higher γ → wider spreads, steeper skew, fewer fills."
    )

    c_s1, c_s2 = st.columns([1, 3])

    with c_s1:
        g1_sym = st.selectbox("Symbol", SYMBOLS, key="g1_sym")
        g1_params = PARAMS[g1_sym]
        g1_T = st.number_input("T (s)", value=3600.0, step=600.0, key="g1_T")
        g1_nsim = st.slider("MC paths", 100, 2000, 400, step=100, key="g1_nsim")
        g1_lo = st.number_input("γ min", value=0.001, format="%.4f", key="g1_lo")
        g1_hi = st.number_input("γ max", value=0.10, format="%.4f", key="g1_hi")
        g1_n = st.slider("# points", 5, 15, 10, key="g1_n")

        run_g1 = st.button("Run γ sweep", type="primary", key="g1_run")

    with c_s2:
        if run_g1:
            gammas = np.geomspace(g1_lo, g1_hi, g1_n)
            N_t_g1 = max(300, int(g1_T))
            results_g1 = []
            prog = st.progress(0, text="Sweeping γ …")

            mid_real_g1 = load_mid_prices(g1_sym)
            mid_arr_g1 = None
            if mid_real_g1 is not None and len(mid_real_g1) > N_t_g1:
                mid_arr_g1 = mid_real_g1.values[:N_t_g1 + 1]

            for idx, g in enumerate(gammas):
                prog.progress((idx + 1) / len(gammas),
                              text=f"γ = {g:.4f}  ({idx+1}/{len(gammas)})")
                try:
                    sol = solve_general(g1_params, g, g1_T, xi=g, N_t=N_t_g1)
                    db0 = sol["delta_bid"][0, :]
                    da0 = sol["delta_ask"][0, :]

                    sp = db0 + da0
                    finite_sp = sp[np.isfinite(sp)]
                    mean_spread = float(np.mean(finite_sp)) if len(finite_sp) > 0 else np.nan

                    sk = db0 - da0
                    lots_s = sol["lots"]
                    mask = np.isfinite(sk)
                    if mask.sum() >= 2:
                        skew_slope = np.polyfit(lots_s[mask], sk[mask], 1)[0]
                    else:
                        skew_slope = np.nan

                    mc = run_quick_mc(g1_params, g, g1_T, N_sim=g1_nsim,
                                      seed=42, mid_prices=mid_arr_g1)

                    results_g1.append({
                        "γ": g, "mean_spread": mean_spread,
                        "skew_slope": skew_slope, **mc,
                    })
                except Exception as e:
                    st.warning(f"γ={g:.4f} failed: {e}")

            prog.empty()
            st.session_state["g1_results"] = pd.DataFrame(results_g1)

        if "g1_results" in st.session_state:
            df = st.session_state["g1_results"]

            fig = make_subplots(
                rows=2, cols=2, vertical_spacing=0.14,
                subplot_titles=[
                    "Mean Spread vs γ", "Skew Slope vs γ",
                    "Fill Rate (fills/h) vs γ", "Sharpe Ratio vs γ",
                ],
            )
            traces = [
                (1, 1, "mean_spread", PALETTE[0]),
                (1, 2, "skew_slope", PALETTE[1]),
                (2, 1, "fill_rate_per_h", PALETTE[2]),
                (2, 2, "sharpe", "#a29bfe"),
            ]
            for r, c, col, color in traces:
                fig.add_trace(go.Scatter(
                    x=df["γ"], y=df[col],
                    mode="lines+markers",
                    marker=dict(size=8, color=color),
                    line=dict(color=color, width=2),
                    showlegend=False,
                ), row=r, col=c)

            for r in (1, 2):
                for c in (1, 2):
                    fig.update_xaxes(title_text="γ", type="log", row=r, col=c)
            fig.update_layout(**PLOT_KW, height=520)
            show(fig)

            st.dataframe(
                df[["γ", "mean_spread", "skew_slope", "fill_rate_per_h",
                    "mean_pnl", "sharpe", "ce"]].style.format({
                    "γ": "{:.4f}", "mean_spread": "{:.4f}",
                    "skew_slope": "{:.4f}", "fill_rate_per_h": "{:.1f}",
                    "mean_pnl": "{:+,.2f}", "sharpe": "{:.3f}",
                    "ce": "{:+,.2f}",
                }),
                use_container_width=True,
            )

            st.info(
                "**Interpretation:** As γ ↑ the market maker widens its spread "
                "(self-protection), steepens the skew (aggressive inventory rebalancing) "
                "and the fill rate drops. The Sharpe typically rises then plateaus — "
                "the model maximises CARA utility, not raw P&L."
            )


# ===========================================================
# TAB 3 — Efficient Frontier  (G2)
# ===========================================================
with tab3:
    st.caption(
        "Each point = optimal market maker at a given γ. "
        "This is the Markowitz-style efficient frontier: "
        "**optimal ≠ highest E[P&L]** — it maximises Certainty Equivalent."
    )

    c_f1, c_f2 = st.columns([1, 3])

    with c_f1:
        ef_sym = st.selectbox("Symbol", SYMBOLS, key="ef_sym")
        ef_params = PARAMS[ef_sym]
        ef_T = st.number_input("T (s)", value=3600.0, step=600.0, key="ef_T")
        ef_nsim = st.slider("MC paths per γ", 200, 3000, 600, step=100, key="ef_nsim")
        ef_lo = st.number_input("γ min", value=0.001, format="%.4f", key="ef_lo")
        ef_hi = st.number_input("γ max", value=0.10, format="%.4f", key="ef_hi")
        ef_n = st.slider("# γ points", 5, 15, 12, key="ef_n")
        run_ef = st.button("Build frontier", type="primary", key="ef_run")

    with c_f2:
        if run_ef:
            gammas_ef = np.geomspace(ef_lo, ef_hi, ef_n)
            N_t_ef = max(300, int(ef_T))

            mid_real_ef = load_mid_prices(ef_sym)
            mid_arr_ef = None
            if mid_real_ef is not None and len(mid_real_ef) > N_t_ef:
                mid_arr_ef = mid_real_ef.values[:N_t_ef + 1]

            frontier = []
            prog = st.progress(0, text="Building efficient frontier …")
            for idx, g in enumerate(gammas_ef):
                prog.progress((idx + 1) / len(gammas_ef),
                              text=f"γ = {g:.4f} — MC …")
                try:
                    mc = run_quick_mc(ef_params, g, ef_T, N_sim=ef_nsim,
                                      seed=42, mid_prices=mid_arr_ef)
                    frontier.append({"γ": g, **mc})
                except Exception as e:
                    st.warning(f"γ={g:.4f} failed: {e}")
            prog.empty()
            st.session_state["ef_results"] = pd.DataFrame(frontier)

        if "ef_results" in st.session_state:
            df = st.session_state["ef_results"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["std_pnl"], y=df["mean_pnl"],
                mode="lines+markers",
                marker=dict(
                    size=14,
                    color=df["sharpe"],
                    colorscale="RdYlGn",
                    colorbar=dict(title="Sharpe", len=0.6),
                    showscale=True,
                    line=dict(width=1, color="rgba(255,255,255,0.3)"),
                ),
                line=dict(width=2, color="rgba(255,255,255,0.15)"),
                text=[
                    f"γ={r['γ']:.4f}<br>"
                    f"Sharpe={r['sharpe']:.3f}<br>"
                    f"CE={r['ce']:+,.2f}<br>"
                    f"VaR 5%={r['var5']:+,.2f}<br>"
                    f"Fills/h={r['fill_rate_per_h']:.1f}"
                    for _, r in df.iterrows()
                ],
                hoverinfo="text",
                name="Frontier",
            ))
            for _, r in df.iterrows():
                fig.add_annotation(
                    x=r["std_pnl"], y=r["mean_pnl"],
                    text=f"γ={r['γ']:.3f}", showarrow=False, yshift=14,
                    font=dict(size=9, color="rgba(255,255,255,0.45)"),
                )
            fig.update_layout(
                **PLOT_KW, height=500,
                xaxis_title="Std[P&L]  (risk)",
                yaxis_title="E[P&L]  (return)",
            )
            show(fig)

            fig2 = make_subplots(
                rows=1, cols=3, horizontal_spacing=0.08,
                subplot_titles=["CE vs γ", "VaR 5% vs γ", "Max Drawdown vs γ"],
            )
            fig2.add_trace(go.Scatter(
                x=df["γ"], y=df["ce"], mode="lines+markers",
                marker=dict(color=PALETTE[0], size=7),
                line=dict(color=PALETTE[0]), showlegend=False,
            ), row=1, col=1)
            fig2.add_trace(go.Scatter(
                x=df["γ"], y=df["var5"], mode="lines+markers",
                marker=dict(color=PALETTE[1], size=7),
                line=dict(color=PALETTE[1]), showlegend=False,
            ), row=1, col=2)
            fig2.add_trace(go.Scatter(
                x=df["γ"], y=df["max_dd"], mode="lines+markers",
                marker=dict(color=PALETTE[2], size=7),
                line=dict(color=PALETTE[2]), showlegend=False,
            ), row=1, col=3)
            for c in (1, 2, 3):
                fig2.update_xaxes(title_text="γ", type="log", row=1, col=c)
            fig2.update_layout(**PLOT_KW, height=300)
            show(fig2)

            st.dataframe(
                df[["γ", "mean_pnl", "std_pnl", "sharpe", "ce",
                    "var5", "max_dd", "fill_rate_per_h", "mean_abs_inv"]].style.format({
                    "γ": "{:.4f}", "mean_pnl": "{:+,.2f}",
                    "std_pnl": "{:,.2f}", "sharpe": "{:.3f}",
                    "ce": "{:+,.2f}", "var5": "{:+,.2f}",
                    "max_dd": "{:,.2f}", "fill_rate_per_h": "{:.1f}",
                    "mean_abs_inv": "{:.2f}",
                }),
                use_container_width=True,
            )

            st.info(
                "**Key insight:** The point with highest E[P&L] is NOT the optimal — "
                "the model maximises **Certainty Equivalent**, which penalises variance. "
                "A naive strategy may beat on raw P&L but lose on risk-adjusted metrics."
            )


# ===========================================================
# TAB 4 — Calibration
# ===========================================================
with tab4:
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
# TAB 5 — Backtest
# ===========================================================
with tab5:
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
            st.session_state["bt_N"] = bt_N
            prog.empty()

        if "bt_results" in st.session_state:
            results = st.session_state["bt_results"]
            bt_N_val = st.session_state.get("bt_N", 500)

            rdf = pd.DataFrame([r.summary() for r in results.values()]).set_index("strategy")
            st.dataframe(rdf.style.format({
                "mean_pnl": "{:+,.2f}", "std_pnl": "{:,.2f}", "sharpe": "{:.3f}",
                "CE": "{:+,.2f}", "mean_fills": "{:.1f}", "mean_abs_inv_T": "{:.2f}",
                "max_drawdown": "{:,.2f}", "pct_flat": "{:.1%}", "mean_fees": "{:,.4f}",
            }), use_container_width=True)

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

            traj = st.slider("Trajectory #", 0, bt_N_val - 1, bt_N_val // 2)
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
# TAB 6 — Intraday Regimes
# ===========================================================
with tab6:
    st.caption(
        "Market conditions change intraday. Compare a **static** policy "
        "(single ODE solve) vs a **regime-aware** one that re-solves at each boundary."
    )

    c_r1, c_r2 = st.columns([1, 3])

    with c_r1:
        rg_sym = st.selectbox("Symbol", SYMBOLS, key="rg_sym")
        rg_params = PARAMS[rg_sym]
        rg_gamma = st.number_input("γ", value=DEFAULT_GAMMA, format="%.4f", key="rg_g")
        rg_T = st.number_input("T (s)", value=3600.0, step=600.0, key="rg_T")
        rg_nsim = st.slider("MC paths", 200, 3000, 600, step=100, key="rg_nsim")

        st.markdown("**Regime multipliers**")
        open_A = st.slider("Open — A ×", 0.5, 2.5, 1.45, 0.05, key="rg_oA")
        open_s = st.slider("Open — σ ×", 0.5, 2.5, 1.50, 0.05, key="rg_os")
        noon_A = st.slider("Noon — A ×", 0.2, 1.5, 0.60, 0.05, key="rg_nA")
        noon_s = st.slider("Noon — σ ×", 0.2, 1.5, 0.55, 0.05, key="rg_ns")
        close_A = st.slider("Close — A ×", 0.5, 2.5, 1.35, 0.05, key="rg_cA")
        close_s = st.slider("Close — σ ×", 0.5, 2.5, 1.40, 0.05, key="rg_cs")

        run_rg = st.button("Run", type="primary", key="rg_run")

    with c_r2:
        if run_rg:
            N_t_rg = max(300, int(rg_T))
            Q_rg = int(rg_params["Q"])
            n_states_rg = 2 * Q_rg + 1
            k_rg = rg_params["k"]
            Delta_rg = rg_params["Delta"]
            dt_rg = rg_T / N_t_rg

            regimes = [
                dict(A_mult=open_A, sigma_mult=open_s, name="Open"),
                dict(A_mult=noon_A, sigma_mult=noon_s, name="Noon"),
                dict(A_mult=close_A, sigma_mult=close_s, name="Close"),
            ]
            breaks = [0, int(0.3 * N_t_rg), int(0.7 * N_t_rg), N_t_rg]

            A_path = np.zeros(N_t_rg)
            sigma_path = np.zeros(N_t_rg)
            for ri, (s, e) in enumerate(zip(breaks[:-1], breaks[1:])):
                A_path[s:e] = rg_params["A"] * regimes[ri]["A_mult"]
                sigma_path[s:e] = rg_params["sigma"] * regimes[ri]["sigma_mult"]

            with st.spinner("Static policy …"):
                sol_static = solve_general(rg_params, rg_gamma, rg_T,
                                           xi=rg_gamma, N_t=N_t_rg)
                db_static = sol_static["delta_bid"]
                da_static = sol_static["delta_ask"]

            with st.spinner("Regime-aware policy …"):
                db_regime = np.full((N_t_rg + 1, n_states_rg), np.nan)
                da_regime = np.full((N_t_rg + 1, n_states_rg), np.nan)
                for ri, (s, e) in enumerate(zip(breaks[:-1], breaks[1:])):
                    rem = N_t_rg - s
                    rem_T = rg_T * rem / N_t_rg
                    p_reg = {
                        **rg_params,
                        "A": rg_params["A"] * regimes[ri]["A_mult"],
                        "sigma": rg_params["sigma"] * regimes[ri]["sigma_mult"],
                    }
                    sol_reg = solve_general(p_reg, rg_gamma, rem_T,
                                            xi=rg_gamma, N_t=rem)
                    seg = e - s
                    db_regime[s:e, :] = sol_reg["delta_bid"][:seg, :]
                    da_regime[s:e, :] = sol_reg["delta_ask"][:seg, :]
                db_regime[-1, :] = db_regime[-2, :]
                da_regime[-1, :] = da_regime[-2, :]

            def _sim_regime(db_tbl, da_tbl):
                rng = np.random.default_rng(42)
                z = rng.standard_normal((rg_nsim, N_t_rg))
                u_b = rng.uniform(size=(rg_nsim, N_t_rg))
                u_a = rng.uniform(size=(rg_nsim, N_t_rg))
                pnl = np.zeros(rg_nsim)
                inv_all = np.zeros((rg_nsim, N_t_rg + 1), dtype=int)
                mtm_all = np.zeros((rg_nsim, N_t_rg + 1))
                fills = np.zeros(rg_nsim, dtype=int)

                for m in range(rg_nsim):
                    S, X, n = 0.0, 0.0, 0
                    for t in range(N_t_rg):
                        il = n + Q_rg
                        if not (0 <= il < n_states_rg):
                            inv_all[m, t + 1] = n
                            mtm_all[m, t + 1] = X + n * Delta_rg * S
                            continue
                        db_v = db_tbl[t, il]
                        da_v = da_tbl[t, il]
                        S += sigma_path[t] * np.sqrt(dt_rg) * z[m, t]
                        if n < Q_rg and np.isfinite(db_v):
                            if u_b[m, t] < A_path[t] * np.exp(-k_rg * db_v) * dt_rg:
                                X -= (S - db_v) * Delta_rg; n += 1; fills[m] += 1
                        if n > -Q_rg and np.isfinite(da_v):
                            if u_a[m, t] < A_path[t] * np.exp(-k_rg * da_v) * dt_rg:
                                X += (S + da_v) * Delta_rg; n -= 1; fills[m] += 1
                        inv_all[m, t + 1] = n
                        mtm_all[m, t + 1] = X + n * Delta_rg * S
                    pnl[m] = X + n * Delta_rg * S
                return pnl, inv_all, mtm_all, fills

            with st.spinner("Simulating static …"):
                pnl_s, inv_s, mtm_s, fills_s = _sim_regime(db_static, da_static)
            with st.spinner("Simulating regime-aware …"):
                pnl_r, inv_r, mtm_r, fills_r = _sim_regime(db_regime, da_regime)

            st.session_state["rg_data"] = dict(
                pnl_s=pnl_s, pnl_r=pnl_r, inv_s=inv_s, inv_r=inv_r,
                mtm_s=mtm_s, mtm_r=mtm_r, fills_s=fills_s, fills_r=fills_r,
                A_path=A_path, sigma_path=sigma_path,
                T=rg_T, N_t=N_t_rg,
            )

        if "rg_data" in st.session_state:
            rd = st.session_state["rg_data"]
            t_ax = np.linspace(0, rd["T"], rd["N_t"] + 1)
            t_ax_inner = np.linspace(0, rd["T"], rd["N_t"])
            C_S, C_R = "#a29bfe", PALETTE[0]

            fig0 = make_subplots(rows=1, cols=2,
                subplot_titles=["A(t)", "σ(t)"])
            fig0.add_trace(go.Scatter(x=t_ax_inner, y=rd["A_path"],
                line=dict(color=C_R, width=2), showlegend=False), row=1, col=1)
            fig0.add_trace(go.Scatter(x=t_ax_inner, y=rd["sigma_path"],
                line=dict(color=C_R, width=2), showlegend=False), row=1, col=2)
            fig0.update_layout(**PLOT_KW, height=220)
            show(fig0)

            def _rg_sum(pnl, fills, label):
                return {
                    "Policy": label,
                    "E[P&L]": np.mean(pnl), "Std[P&L]": np.std(pnl),
                    "Sharpe": np.mean(pnl) / np.std(pnl) if np.std(pnl) > 0 else 0,
                    "VaR 5%": np.percentile(pnl, 5),
                    "Mean fills": np.mean(fills),
                }
            st.dataframe(
                pd.DataFrame([
                    _rg_sum(rd["pnl_s"], rd["fills_s"], "Static"),
                    _rg_sum(rd["pnl_r"], rd["fills_r"], "Regime-aware"),
                ]).set_index("Policy").style.format({
                    "E[P&L]": "{:+,.2f}", "Std[P&L]": "{:,.2f}",
                    "Sharpe": "{:.3f}", "VaR 5%": "{:+,.2f}",
                    "Mean fills": "{:.1f}",
                }),
                use_container_width=True,
            )

            fig1 = make_subplots(rows=1, cols=2,
                subplot_titles=["P&L distribution", "Average MtM"])
            fig1.add_trace(go.Histogram(x=rd["pnl_s"], name="Static",
                opacity=0.5, marker_color=C_S, nbinsx=60), row=1, col=1)
            fig1.add_trace(go.Histogram(x=rd["pnl_r"], name="Regime",
                opacity=0.5, marker_color=C_R, nbinsx=60), row=1, col=1)
            fig1.add_trace(go.Scatter(x=t_ax, y=np.mean(rd["mtm_s"], axis=0),
                name="Static", line=dict(color=C_S, width=2)), row=1, col=2)
            fig1.add_trace(go.Scatter(x=t_ax, y=np.mean(rd["mtm_r"], axis=0),
                name="Regime", line=dict(color=C_R, width=2)), row=1, col=2)
            fig1.update_layout(**PLOT_KW, height=360, barmode="overlay")
            show(fig1)

            step_r = max(1, rd["inv_s"].shape[1] // 500)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=t_ax[::step_r],
                y=np.mean(np.abs(rd["inv_s"][:, ::step_r].astype(float)), axis=0),
                name="Static", line=dict(color=C_S, width=2)))
            fig2.add_trace(go.Scatter(
                x=t_ax[::step_r],
                y=np.mean(np.abs(rd["inv_r"][:, ::step_r].astype(float)), axis=0),
                name="Regime", line=dict(color=C_R, width=2)))
            fig2.update_layout(**PLOT_KW, height=300,
                xaxis_title="t (s)", yaxis_title="E[|inventory|]")
            show(fig2)

            st.info(
                "**Interpretation:** The regime-aware policy adapts quotes to changing "
                "conditions — wider spreads when volatile (open/close), tighter at noon. "
                "This typically yields better Sharpe than the static policy."
            )


# ===========================================================
# TAB 7 — Hawkes vs Poisson
# ===========================================================
with tab7:
    if not HAWKES_AVAILABLE:
        st.warning("Hawkes module not found — ensure `src/intensity_hawkes.py` exists.")
    else:
        st.caption(
            "Same ODE-optimal quotes, two fill engines. "
            "Hawkes adds self- and cross-excitation → fill clustering, "
            "higher variance, and breaks the Guéant dimensional reduction."
        )

        c_h1, c_h2 = st.columns([1, 3])

        with c_h1:
            hk_sym = st.selectbox("Symbol", SYMBOLS, key="hk_sym")
            hk_params = PARAMS[hk_sym]
            hk_gamma = st.number_input("γ", value=DEFAULT_GAMMA, format="%.4f", key="hk_g")
            hk_T = st.number_input("T (s)", value=3600.0, step=600.0, key="hk_T")
            hk_nsim = st.slider("MC paths", 200, 3000, 600, step=100, key="hk_nsim")

            st.markdown("**Hawkes parameters**")
            hk_beta = st.number_input("β (decay)", value=10.0, step=1.0, key="hk_b")
            hk_as = st.number_input("α_self", value=2.0, step=0.5, key="hk_as")
            hk_ac = st.number_input("α_cross", value=0.5, step=0.1, key="hk_ac")

            run_hk = st.button("Run", type="primary", key="hk_run")

        with c_h2:
            if run_hk:
                N_t_hk = max(300, int(hk_T))
                dt_hk = hk_T / N_t_hk
                Q_hk = int(hk_params["Q"])
                ns_hk = 2 * Q_hk + 1
                A_hk = hk_params["A"]
                k_hk = hk_params["k"]
                Delta_hk = hk_params["Delta"]
                sigma_hk = hk_params["sigma"]
                hawkes_cfg = {"beta": hk_beta, "alpha_self": hk_as, "alpha_cross": hk_ac}

                with st.spinner("Solving ODE …"):
                    sol_hk = solve_general(hk_params, hk_gamma, hk_T,
                                           xi=hk_gamma, N_t=N_t_hk)
                    db_hk = sol_hk["delta_bid"]
                    da_hk = sol_hk["delta_ask"]

                seed_hk = 42
                rng_p = np.random.default_rng(seed_hk)
                z_hk = rng_p.standard_normal((hk_nsim, N_t_hk))
                ub_hk = rng_p.uniform(size=(hk_nsim, N_t_hk))
                ua_hk = rng_p.uniform(size=(hk_nsim, N_t_hk))

                def _get_quotes(t, n):
                    il = n + Q_hk
                    if not (0 <= il < ns_hk):
                        return np.inf, np.inf
                    d_b = db_hk[t, il] if (n < Q_hk and np.isfinite(db_hk[t, il])) else np.inf
                    d_a = da_hk[t, il] if (n > -Q_hk and np.isfinite(da_hk[t, il])) else np.inf
                    return d_b, d_a

                with st.spinner("Simulating Poisson …"):
                    pnl_P = np.zeros(hk_nsim)
                    fills_P = np.zeros(hk_nsim, dtype=int)
                    inv_P = np.zeros((hk_nsim, N_t_hk + 1), dtype=int)

                    for m in range(hk_nsim):
                        S, X, n = 0.0, 0.0, 0
                        for t in range(N_t_hk):
                            d_b, d_a = _get_quotes(t, n)
                            S += sigma_hk * np.sqrt(dt_hk) * z_hk[m, t]
                            if d_b < np.inf:
                                lam = A_hk * np.exp(-k_hk * max(d_b, 0))
                                if ub_hk[m, t] < 1 - np.exp(-lam * dt_hk):
                                    X -= (S - d_b) * Delta_hk; n += 1; fills_P[m] += 1
                            if d_a < np.inf:
                                lam = A_hk * np.exp(-k_hk * max(d_a, 0))
                                if ua_hk[m, t] < 1 - np.exp(-lam * dt_hk):
                                    X += (S + d_a) * Delta_hk; n -= 1; fills_P[m] += 1
                            inv_P[m, t + 1] = n
                        pnl_P[m] = X + n * Delta_hk * S

                with st.spinner("Simulating Hawkes …"):
                    rng_h = np.random.default_rng(seed_hk + 1)
                    ub_H = rng_h.uniform(size=(hk_nsim, N_t_hk))
                    ua_H = rng_h.uniform(size=(hk_nsim, N_t_hk))

                    pnl_H = np.zeros(hk_nsim)
                    fills_H = np.zeros(hk_nsim, dtype=int)
                    inv_H = np.zeros((hk_nsim, N_t_hk + 1), dtype=int)
                    exc_hist = np.zeros((hk_nsim, N_t_hk))

                    for m in range(hk_nsim):
                        S, X, n = 0.0, 0.0, 0
                        hs = HawkesState(hawkes_cfg)
                        for t in range(N_t_hk):
                            d_b, d_a = _get_quotes(t, n)
                            S += sigma_hk * np.sqrt(dt_hk) * z_hk[m, t]

                            mu_b = A_hk * np.exp(-k_hk * max(d_b, 0)) if d_b < np.inf else 0.0
                            mu_a = A_hk * np.exp(-k_hk * max(d_a, 0)) if d_a < np.inf else 0.0

                            lam_b = hs.lambda_bid(mu_b)
                            lam_a = hs.lambda_ask(mu_a)

                            fb = (ub_H[m, t] < fill_prob_from_intensity(lam_b, dt_hk)) if d_b < np.inf else False
                            fa = (ua_H[m, t] < fill_prob_from_intensity(lam_a, dt_hk)) if d_a < np.inf else False

                            if fb:
                                X -= (S - d_b) * Delta_hk; n += 1; fills_H[m] += 1
                            if fa:
                                X += (S + d_a) * Delta_hk; n -= 1; fills_H[m] += 1

                            hs.step(dt_hk, fb, fa)
                            exc_hist[m, t] = hs.y_bid + hs.y_ask
                            inv_H[m, t + 1] = n
                        pnl_H[m] = X + n * Delta_hk * S

                st.session_state["hk_data"] = dict(
                    pnl_P=pnl_P, pnl_H=pnl_H,
                    fills_P=fills_P, fills_H=fills_H,
                    inv_P=inv_P, inv_H=inv_H,
                    exc_hist=exc_hist,
                    T=hk_T, N_t=N_t_hk,
                )

            if "hk_data" in st.session_state:
                hd = st.session_state["hk_data"]
                t_ax_h = np.linspace(0, hd["T"], hd["N_t"] + 1)
                C_P, C_H = "#74b9ff", "#ff7675"

                def _hk_sum(pnl, fills, label):
                    return {
                        "Model": label,
                        "E[P&L]": np.mean(pnl), "Std[P&L]": np.std(pnl),
                        "Sharpe": np.mean(pnl) / np.std(pnl) if np.std(pnl) > 0 else 0,
                        "Mean fills": np.mean(fills),
                        "Fills/h": np.mean(fills) / max(hd["T"] / 3600, 1e-6),
                    }
                st.dataframe(
                    pd.DataFrame([
                        _hk_sum(hd["pnl_P"], hd["fills_P"], "Poisson"),
                        _hk_sum(hd["pnl_H"], hd["fills_H"], "Hawkes"),
                    ]).set_index("Model").style.format({
                        "E[P&L]": "{:+,.2f}", "Std[P&L]": "{:,.2f}",
                        "Sharpe": "{:.3f}", "Mean fills": "{:.1f}",
                        "Fills/h": "{:.1f}",
                    }),
                    use_container_width=True,
                )

                fig = make_subplots(
                    rows=2, cols=2, vertical_spacing=0.14,
                    subplot_titles=[
                        "P&L distribution", "Fill count distribution",
                        "E[|inventory|]", "Mean excitation state",
                    ],
                )
                fig.add_trace(go.Histogram(x=hd["pnl_P"], name="Poisson",
                    opacity=0.5, marker_color=C_P, nbinsx=60), row=1, col=1)
                fig.add_trace(go.Histogram(x=hd["pnl_H"], name="Hawkes",
                    opacity=0.5, marker_color=C_H, nbinsx=60), row=1, col=1)

                fig.add_trace(go.Histogram(x=hd["fills_P"], name="Poisson",
                    opacity=0.5, marker_color=C_P, showlegend=False), row=1, col=2)
                fig.add_trace(go.Histogram(x=hd["fills_H"], name="Hawkes",
                    opacity=0.5, marker_color=C_H, showlegend=False), row=1, col=2)

                step_h = max(1, hd["inv_P"].shape[1] // 500)
                fig.add_trace(go.Scatter(
                    x=t_ax_h[::step_h],
                    y=np.mean(np.abs(hd["inv_P"][:, ::step_h].astype(float)), axis=0),
                    name="Poisson", line=dict(color=C_P, width=2), showlegend=False,
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=t_ax_h[::step_h],
                    y=np.mean(np.abs(hd["inv_H"][:, ::step_h].astype(float)), axis=0),
                    name="Hawkes", line=dict(color=C_H, width=2), showlegend=False,
                ), row=2, col=1)

                step_e = max(1, hd["exc_hist"].shape[1] // 500)
                mean_exc = np.mean(hd["exc_hist"][:, ::step_e], axis=0)
                fig.add_trace(go.Scatter(
                    x=np.linspace(0, hd["T"], len(mean_exc)),
                    y=mean_exc, line=dict(color=C_H, width=2),
                    showlegend=False,
                ), row=2, col=2)

                fig.update_layout(**PLOT_KW, height=560, barmode="overlay")
                show(fig)

                st.info(
                    "**Key finding:** Hawkes fills introduce clustering — bursts of fills "
                    "then quiet periods. This increases P&L variance. Crucially, the Hawkes "
                    "dynamics **destroy the dimensional reduction** central to the Guéant "
                    "framework: the Hamiltonian is no longer expressible in closed form, "
                    "so ODE-optimal quotes are no longer truly optimal under Hawkes."
                )


# ===========================================================
# TAB 8 — Paper Trading
# ===========================================================
with tab8:
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