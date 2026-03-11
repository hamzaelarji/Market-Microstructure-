"""ui/tabs/sensitivity.py — Sensitivity Atlas: 1D sweep + 2D heatmaps."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_making.core.solver_1d import solve_general
from market_making.simulation.backtest import BacktestConfig, run_backtest
from ui.styles import TEAL, RED, YELLOW, PURPLE, BLUE, ORANGE, GRAY, hero_banner, section_header

# ─── Parameter meta ──────────────────────────────────────────────────────────

PARAM_META = {
    "γ (risk aversion)":  {"key": "gamma",  "scale": "log",   "base_key": None},
    "σ (volatility)":     {"key": "sigma",  "scale": "linear","base_key": "sigma"},
    "k (order decay)":    {"key": "k",      "scale": "log",   "base_key": "k"},
    "A (arrival rate)":   {"key": "A",      "scale": "log",   "base_key": "A"},
    "T (horizon, s)":     {"key": "T",      "scale": "linear","base_key": None},
}

METRIC_META = {
    "Sharpe Ratio":     "sharpe",
    "Mean P&L ($)":     "mean_pnl",
    "VaR 5% ($)":       "var5",
    "Fill Rate (fills/h)": "fill_rate_per_h",
    "Mean Spread ($)":  "mean_spread",
}

COLORS = [TEAL, RED, YELLOW, PURPLE, BLUE, ORANGE]


def _run_grid_point(params_base: dict, gamma_base: float, T_base: float,
                    param_x_key: str, vx: float,
                    param_y_key: str, vy: float,
                    N_sim: int = 80, seed: int = 42) -> dict:
    """Run one backtest on the sensitivity grid."""
    p = params_base.copy()
    gamma = gamma_base
    T = T_base

    for key, val in [(param_x_key, vx), (param_y_key, vy)]:
        if key == "gamma":
            gamma = val
        elif key == "T":
            T = val
        elif key in p:
            p[key] = val

    N_t = max(300, int(T))
    cfg = BacktestConfig(params=p, gamma=gamma, T=T, N_t=N_t,
                         N_sim=N_sim, seed=seed, strategy="optimal")
    try:
        res = run_backtest(cfg)
        pnl = res.pnl
        var5 = float(np.percentile(pnl, 5))

        # Compute mean spread from ODE at t=0, n=0
        sol = solve_general(p, gamma, T, xi=gamma, N_t=max(50, int(T // 60)))
        Q = int(p["Q"])
        db_mid = sol["delta_bid"][0, Q]  # n=0 → index Q
        da_mid = sol["delta_ask"][0, Q]
        mean_spread = float(db_mid + da_mid) if (np.isfinite(db_mid) and np.isfinite(da_mid)) else np.nan

        return {
            "mean_pnl": float(np.mean(pnl)),
            "std_pnl":  float(np.std(pnl)),
            "sharpe":   float(np.mean(pnl) / np.std(pnl)) if np.std(pnl) > 1e-10 else 0.0,
            "var5":     var5,
            "fill_rate_per_h": float(np.mean(res.n_bid_fills + res.n_ask_fills)) / max(T / 3600, 1e-6),
            "mean_spread": mean_spread,
        }
    except Exception:
        return {k: np.nan for k in ["mean_pnl", "std_pnl", "sharpe", "var5",
                                     "fill_rate_per_h", "mean_spread"]}


def _param_range(label: str, meta: dict, base_val: float,
                 lo_mul: float = 0.1, hi_mul: float = 5.0, n: int = 8) -> np.ndarray:
    if meta["scale"] == "log":
        return np.geomspace(base_val * lo_mul, base_val * hi_mul, n)
    else:
        return np.linspace(base_val * lo_mul, base_val * hi_mul, n)


# ─── 1D sweep ─────────────────────────────────────────────────────────────────

def _render_1d_sweep(PARAMS, PLOT_KW, show):
    section_header("1D Sensitivity Sweep")
    SYMBOLS = list(PARAMS.keys())

    c1, c2 = st.columns([1, 3], gap="large")
    with c1:
        sym    = st.selectbox("Symbol", SYMBOLS, key="g1_sym")
        params = PARAMS[sym]
        T      = st.number_input("T (s)", value=3600.0, key="g1_T")
        nsim   = st.slider("MC paths", 100, 1000, 300, step=100, key="g1_nsim")
        lo     = st.number_input("γ min", value=0.001, format="%.4f", key="g1_lo")
        hi     = st.number_input("γ max", value=0.10, format="%.4f", key="g1_hi")
        n_pts  = st.slider("# points", 5, 15, 10, key="g1_n")
        run_1d = st.button("Run γ sweep", type="primary", key="g1_run",
                           use_container_width=True)

    with c2:
        if run_1d:
            gammas = np.geomspace(lo, hi, n_pts)
            rows = []
            prog = st.progress(0, text="Sweeping γ …")
            for idx, g in enumerate(gammas):
                prog.progress((idx + 1) / len(gammas), text=f"γ = {g:.4f}")
                try:
                    N_t = max(300, int(T))
                    sol = solve_general(params, g, T, xi=g, N_t=N_t)
                    db0 = sol["delta_bid"][0, :]
                    da0 = sol["delta_ask"][0, :]
                    sp = db0 + da0
                    sp_finite = sp[np.isfinite(sp)]
                    mean_spread = float(np.mean(sp_finite)) if len(sp_finite) else np.nan

                    sk = db0 - da0
                    lots = sol["lots"]
                    mask = np.isfinite(sk)
                    skew_slope = float(np.polyfit(lots[mask], sk[mask], 1)[0]) if mask.sum() >= 2 else np.nan

                    cfg = BacktestConfig(params=params, gamma=g, T=T, N_t=N_t,
                                        N_sim=nsim, seed=42, strategy="optimal")
                    res = run_backtest(cfg)
                    sharpe = res.sharpe
                    fill_rate = float(np.mean(res.n_bid_fills + res.n_ask_fills)) / max(T / 3600, 1e-6)
                    mean_pnl = float(np.mean(res.pnl))

                    rows.append({"γ": g, "mean_spread": mean_spread,
                                 "skew_slope": skew_slope, "sharpe": sharpe,
                                 "fill_rate_per_h": fill_rate, "mean_pnl": mean_pnl})
                except Exception as e:
                    st.warning(f"γ={g:.4f} failed: {e}")
            prog.empty()
            st.session_state["g1_results"] = pd.DataFrame(rows)

        if "g1_results" in st.session_state:
            df = st.session_state["g1_results"]
            traces = [
                (1, 1, "mean_spread", TEAL, "Mean Spread vs γ"),
                (1, 2, "skew_slope", RED, "Skew Slope vs γ"),
                (2, 1, "fill_rate_per_h", YELLOW, "Fill Rate (fills/h) vs γ"),
                (2, 2, "sharpe", PURPLE, "Sharpe Ratio vs γ"),
            ]
            fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14,
                                subplot_titles=[t[4] for t in traces])
            for r, c, col, color, _ in traces:
                fig.add_trace(go.Scatter(
                    x=df["γ"], y=df[col], mode="lines+markers",
                    marker=dict(size=8, color=color),
                    line=dict(color=color, width=2),
                    showlegend=False,
                ), row=r, col=c)
                for rc in [(r, c)]:
                    fig.update_xaxes(title_text="γ", type="log", row=r, col=c)
            fig.update_layout(**PLOT_KW, height=520)
            show(fig)
            st.dataframe(
                df[["γ", "mean_spread", "skew_slope", "fill_rate_per_h",
                    "mean_pnl", "sharpe"]].style.format({
                    "γ": "{:.4f}", "mean_spread": "{:.4f}", "skew_slope": "{:.4f}",
                    "fill_rate_per_h": "{:.1f}", "mean_pnl": "{:+,.2f}", "sharpe": "{:.3f}",
                }),
                use_container_width=True,
            )
            st.info(
                "**Interpretation:** As γ ↑ the market maker widens its spread "
                "(self-protection), steepens the skew (aggressive inventory rebalancing) "
                "and the fill rate drops. Sharpe typically rises then plateaus — "
                "the model maximises CARA utility, not raw P&L."
            )


# ─── 2D heatmap ───────────────────────────────────────────────────────────────

def _render_2d_heatmap(PARAMS, PLOT_KW, show):
    section_header("2D Sensitivity Heatmap")
    SYMBOLS = list(PARAMS.keys())

    c1, c2 = st.columns([1, 3], gap="large")
    with c1:
        sym    = st.selectbox("Symbol", SYMBOLS, key="h2_sym")
        params = PARAMS[sym].copy()
        gamma0 = st.number_input("Base γ", value=0.01, format="%.4f", key="h2_g0")
        T0     = st.number_input("Base T (s)", value=3600.0, key="h2_T0")
        nsim   = st.slider("MC paths (per point)", 50, 300, 80, step=50, key="h2_nsim")

        param_labels = list(PARAM_META.keys())
        px_label = st.selectbox("X-axis parameter", param_labels,
                                index=0, key="h2_px")
        py_label = st.selectbox("Y-axis parameter", param_labels,
                                index=1, key="h2_py")
        metric_label = st.selectbox("Metric", list(METRIC_META.keys()), key="h2_met")

        resolution = st.slider("Grid resolution (per axis)", 5, 12, 7, key="h2_res")

        st.caption(
            f"Grid: {resolution}×{resolution} = **{resolution**2} runs**  \n"
            f"≈ {resolution**2 * nsim // 1000 + 1}–{resolution**2 * nsim // 500 + 2}s compute"
        )

        run_h2 = st.button("🌡️ Compute Heatmap", type="primary", key="h2_run",
                            use_container_width=True)

    with c2:
        if run_h2:
            pmx = PARAM_META[px_label]
            pmy = PARAM_META[py_label]
            metric_key = METRIC_META[metric_label]

            # Base values for each param
            base_x = gamma0 if pmx["key"] == "gamma" else (T0 if pmx["key"] == "T" else params.get(pmx["key"], 1.0))
            base_y = gamma0 if pmy["key"] == "gamma" else (T0 if pmy["key"] == "T" else params.get(pmy["key"], 1.0))

            xs = _param_range(px_label, pmx, base_x, n=resolution)
            ys = _param_range(py_label, pmy, base_y, n=resolution)

            total = resolution * resolution
            grid  = np.full((resolution, resolution), np.nan)
            prog  = st.progress(0, text="Computing grid …")

            for i, vx in enumerate(xs):
                for j, vy in enumerate(ys):
                    idx_flat = i * resolution + j
                    prog.progress((idx_flat + 1) / total,
                                  text=f"Point {idx_flat+1}/{total}")
                    pt = _run_grid_point(
                        params, gamma0, T0,
                        pmx["key"], vx, pmy["key"], vy, nsim,
                    )
                    grid[j, i] = pt.get(metric_key, np.nan)

            prog.empty()
            st.session_state["h2_result"] = {
                "grid": grid, "xs": xs, "ys": ys,
                "px_label": px_label, "py_label": py_label,
                "metric_label": metric_label,
                "px_scale": pmx["scale"], "py_scale": pmy["scale"],
            }

        if "h2_result" in st.session_state:
            r = st.session_state["h2_result"]
            grid = r["grid"]
            xs   = r["xs"]
            ys   = r["ys"]

            # Format axis tick labels nicely
            def _fmt(v, scale):
                if scale == "log":
                    return f"{v:.2e}"
                return f"{v:.3g}"

            x_labels = [_fmt(v, r["px_scale"]) for v in xs]
            y_labels = [_fmt(v, r["py_scale"]) for v in ys]

            # Colorscale selection
            metric_key = METRIC_META.get(r["metric_label"], "sharpe")
            if "pnl" in metric_key or "sharpe" in metric_key:
                colorscale = "RdYlGn"
            elif "var" in metric_key:
                colorscale = [[0, "#003d2e"], [0.5, "#1a1d24"], [1, "#3d0a0a"]]
            else:
                colorscale = "Viridis"

            # Build text matrix
            text_fmt = [[f"{grid[j, i]:.3g}" if np.isfinite(grid[j, i]) else "N/A"
                         for i in range(len(xs))]
                        for j in range(len(ys))]

            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=grid, x=x_labels, y=y_labels,
                text=text_fmt,
                texttemplate="%{text}",
                textfont=dict(size=9, color="white"),
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title=r["metric_label"], thickness=14, len=0.9),
            ))

            # Contour lines overlay
            finite_grid = grid.copy()
            finite_grid[~np.isfinite(finite_grid)] = np.nanmean(grid)
            fig.add_trace(go.Contour(
                z=finite_grid, x=x_labels, y=y_labels,
                showscale=False,
                contours=dict(showlabels=False, coloring="none"),
                line=dict(color="rgba(255,255,255,0.3)", width=0.8),
                ncontours=8,
            ))

            fig.update_layout(
                **PLOT_KW, height=520,
                xaxis_title=r["px_label"],
                yaxis_title=r["py_label"],
                title=dict(
                    text=f"<b>{r['metric_label']}</b> as function of "
                         f"{r['px_label']} (x) × {r['py_label']} (y)",
                    font=dict(size=13), x=0,
                ),
            )
            show(fig)

            st.caption(
                "**Contour lines** highlight iso-value curves.  "
                f"Each cell = {METRIC_META[r['metric_label']]} "
                f"from optimal market making (N_sim={nsim} paths per point)."
            )


# ─── Main render ──────────────────────────────────────────────────────────────

def render_sensitivity(PARAMS, META, HAS_REAL_DATA, PALETTE, PLOT_KW, show,
                       load_mid_prices, run_quick_mc):
    """Render Sensitivity Atlas — 1D sweep + 2D heatmap."""

    hero_banner(
        "🌡️",
        "Sensitivity Atlas",
        "Explore how any parameter pair affects spread, P&L, Sharpe, and fill rate.",
    )

    tab1d, tab2d = st.tabs(["📈  1D γ Sweep", "🌡️  2D Heatmap"])

    with tab1d:
        _render_1d_sweep(PARAMS, PLOT_KW, show)

    with tab2d:
        _render_2d_heatmap(PARAMS, PLOT_KW, show)
