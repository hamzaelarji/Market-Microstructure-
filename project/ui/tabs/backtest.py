"""ui/tabs/backtest.py — Strategy Battle: fan charts, leaderboard, dominance matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_making.simulation.backtest import BacktestConfig, run_backtest
from ui.styles import (
    TEAL, RED, YELLOW, PURPLE, ORANGE, GRAY,
    hero_banner, section_header, strategy_color,
)

DEFAULT_GAMMA = 0.01

STRATEGIES = {
    "optimal":       "Optimal (Guéant ODE)",
    "naive":         "Naïve (fixed spread)",
    "closed_form":   "Closed-form approx.",
}


def _run_all(params, gamma, T, N_sim, fee, strats, mid_prices_arr, progress):
    results = {}
    N_t = max(600, int(T))
    for i, strat in enumerate(strats):
        progress.progress((i) / len(strats), text=f"Running {strat} …")
        cfg = BacktestConfig(
            params=params, gamma=gamma, T=T, N_t=N_t,
            N_sim=N_sim, maker_fee=fee, strategy=strat, seed=42,
            mid_prices=mid_prices_arr,
        )
        results[strat] = run_backtest(cfg)
    progress.progress(1.0, text="Done.")
    return results


def _fan_chart(results: dict, PLOT_KW: dict) -> go.Figure:
    """P&L fan chart: percentile bands over time for each strategy."""
    fig = go.Figure()

    PCTS = [5, 25, 50, 75, 95]

    for strat, res in results.items():
        color = strategy_color(strat)
        pcts = np.percentile(res.mtm, PCTS, axis=0)    # (5, N_t+1)
        times = res.times

        label = STRATEGIES.get(strat, strat)

        # Shaded band P25–P75
        r, g, b = _hex_to_rgb(color)
        fig.add_trace(go.Scatter(
            x=np.concatenate([times, times[::-1]]),
            y=np.concatenate([pcts[3], pcts[1][::-1]]),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.12)",
            line=dict(width=0),
            showlegend=False,
            name=f"{label} P25–P75",
        ))

        # Outer band P5–P95
        fig.add_trace(go.Scatter(
            x=np.concatenate([times, times[::-1]]),
            y=np.concatenate([pcts[4], pcts[0][::-1]]),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.05)",
            line=dict(width=0),
            showlegend=False,
            name=f"{label} P5–P95",
        ))

        # Median line
        fig.add_trace(go.Scatter(
            x=times, y=pcts[2],
            name=label,
            mode="lines",
            line=dict(color=color, width=2.5),
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.15)")

    fig.update_layout(
        **PLOT_KW,
        height=380,
        xaxis_title="Time (s)",
        yaxis_title="MtM P&L ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=10)),
        title=dict(
            text="P&L Fan Chart  <span style='font-size:11px;color:#555e6e'>"
                 "(solid = median, bands = P25–P75 and P5–P95)</span>",
            font=dict(size=13), x=0,
        ),
    )
    return fig


def _inventory_chart(results: dict, PLOT_KW: dict) -> go.Figure:
    """Mean |inventory| over time for each strategy."""
    fig = go.Figure()
    for strat, res in results.items():
        color = strategy_color(strat)
        step = max(1, res.inventory.shape[1] // 400)
        mean_abs = np.mean(np.abs(res.inventory[:, ::step].astype(float)), axis=0)
        fig.add_trace(go.Scatter(
            x=res.times[::step], y=mean_abs,
            name=STRATEGIES.get(strat, strat),
            mode="lines", line=dict(color=color, width=2),
        ))
    fig.update_layout(
        **PLOT_KW, height=280,
        xaxis_title="Time (s)", yaxis_title="E[|inventory|] (lots)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=10)),
        title=dict(text="Mean Absolute Inventory", font=dict(size=13), x=0),
    )
    return fig


def _pnl_dist_chart(results: dict, PLOT_KW: dict) -> go.Figure:
    """Terminal P&L distribution (histogram KDE-style)."""
    fig = go.Figure()
    for strat, res in results.items():
        color = strategy_color(strat)
        r, g, b = _hex_to_rgb(color)
        fig.add_trace(go.Histogram(
            x=res.pnl, name=STRATEGIES.get(strat, strat),
            opacity=0.6, nbinsx=60,
            marker=dict(color=f"rgba({r},{g},{b},0.7)",
                        line=dict(width=0.5, color=color)),
        ))
        fig.add_vline(
            x=float(np.mean(res.pnl)),
            line_dash="dash", line_color=color, line_width=1.5,
            annotation_text=f"μ={np.mean(res.pnl):+,.1f}",
            annotation_font=dict(size=9, color=color),
        )
    fig.update_layout(
        **PLOT_KW, height=280, barmode="overlay",
        xaxis_title="Terminal P&L ($)", yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=10)),
        title=dict(text="Terminal P&L Distribution", font=dict(size=13), x=0),
    )
    return fig


def _dominance_chart(results: dict, PLOT_KW: dict) -> go.Figure:
    """Dominance matrix: P(strat_i beats strat_j) as a heatmap."""
    strats = list(results.keys())
    n = len(strats)
    mat = np.zeros((n, n))
    for i, sa in enumerate(strats):
        for j, sb in enumerate(strats):
            if i == j:
                mat[i, j] = 0.5
            else:
                mat[i, j] = float(np.mean(results[sa].pnl > results[sb].pnl))

    labels = [STRATEGIES.get(s, s) for s in strats]
    text = [[f"{mat[i,j]:.0%}" for j in range(n)] for i in range(n)]

    fig = go.Figure(go.Heatmap(
        z=mat, x=labels, y=labels,
        text=text, texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        colorscale=[[0, "#1a0a0a"], [0.5, "#1a1d24"], [1, "#003d2e"]],
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(title="P(row beats col)", tickformat=".0%",
                      len=0.8, thickness=12),
    ))
    fig.update_layout(
        **PLOT_KW, height=280,
        xaxis_title="Opponent", yaxis_title="Strategy",
        title=dict(text="Dominance Matrix  "
                   "<span style='font-size:11px;color:#555e6e'>"
                   "P(row beats column in P&L)</span>",
                   font=dict(size=13), x=0),
    )
    return fig


def _leaderboard(results: dict) -> pd.DataFrame:
    rows = []
    for strat, res in results.items():
        pnl = res.pnl
        var5 = float(np.percentile(pnl, 5))
        rows.append({
            "Strategy": STRATEGIES.get(strat, strat),
            "Mean P&L": res.mean_pnl,
            "Std P&L": res.std_pnl,
            "Sharpe": res.sharpe,
            "VaR 5%": var5,
            "Max DD": res.max_drawdown,
            "Fill Rate (fills)": res.mean_fills,
            "E[|inv|] final": res.mean_abs_inventory,
        })
    return pd.DataFrame(rows).set_index("Strategy")


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ─── Main render ──────────────────────────────────────────────────────────────

def render_backtest(PARAMS, META, HAS_REAL_DATA, PALETTE, PLOT_KW, show,
                    load_mid_prices, run_quick_mc):
    """Render Strategy Battle tab — fan charts, leaderboard, dominance."""

    hero_banner(
        "⚔️",
        "Strategy Battle",
        "Fan charts, leaderboard & dominance analysis across market-making strategies.",
    )

    SYMBOLS = list(PARAMS.keys())
    c_set, c_res = st.columns([1, 3], gap="large")

    with c_set:
        section_header("Setup")

        sym = st.selectbox("Symbol", SYMBOLS, key="bt_sym")
        bt_params = PARAMS[sym].copy()

        gamma   = st.number_input("γ", value=DEFAULT_GAMMA, format="%.4f", key="bt_g")
        T       = st.number_input("T (s)", value=3600.0, key="bt_t")
        N_sim   = st.slider("Simulations (N_sim)", 200, 2000, 500, step=100)
        fee_bps = st.number_input("Maker fee (bps)", value=1.0, step=0.5)
        fee     = fee_bps / 10_000

        mid_real = load_mid_prices(sym)
        use_real = False
        if mid_real is not None:
            use_real = st.checkbox("Use real price path", value=True)

        st.divider()
        section_header("Strategies")

        sel_strats = st.multiselect(
            "Strategies to compare",
            list(STRATEGIES.keys()),
            default=["optimal", "naive"],
            format_func=lambda k: STRATEGIES[k],
        )

        run_btn = st.button("▶ Run Battle", type="primary", key="bt_run",
                            use_container_width=True)

    with c_res:
        if run_btn and sel_strats:
            N_t = max(600, int(T))
            mid_arr = None
            if use_real and mid_real is not None and len(mid_real) > N_t + 1:
                idx0 = np.random.randint(0, len(mid_real) - N_t - 1)
                mid_arr = mid_real.values[idx0:idx0 + N_t + 1]

            prog = st.progress(0)
            with st.spinner("Running Monte Carlo backtest …"):
                results = _run_all(bt_params, gamma, T, N_sim, fee,
                                   sel_strats, mid_arr, prog)
            prog.empty()
            st.session_state["bt_results"] = results

        if "bt_results" not in st.session_state:
            st.info("Configure strategies on the left and click **▶ Run Battle**.")
            return

        results = st.session_state["bt_results"]
        if not results:
            return

        # ── Leaderboard ─────────────────────────────────────────────────────
        section_header("Leaderboard")
        lb = _leaderboard(results)

        def _color_sharpe(v):
            if v > 0.5:
                return "color: #00d4aa"
            elif v < 0:
                return "color: #ff6b6b"
            return ""

        st.dataframe(
            lb.style
            .format({
                "Mean P&L": "{:+,.2f}",
                "Std P&L":  "{:,.2f}",
                "Sharpe":   "{:.3f}",
                "VaR 5%":   "{:+,.2f}",
                "Max DD":   "{:,.2f}",
                "Fill Rate (fills)": "{:.1f}",
                "E[|inv|] final":    "{:.2f}",
            })
            .applymap(_color_sharpe, subset=["Sharpe"]),
            use_container_width=True,
        )

        # ── Fan chart ────────────────────────────────────────────────────────
        section_header("P&L Fan Chart")
        show(_fan_chart(results, PLOT_KW))

        # ── P&L distribution + inventory side by side ────────────────────────
        col_d, col_i = st.columns(2, gap="medium")
        with col_d:
            section_header("Terminal P&L Distribution")
            show(_pnl_dist_chart(results, PLOT_KW))
        with col_i:
            section_header("Mean Absolute Inventory")
            show(_inventory_chart(results, PLOT_KW))

        # ── Dominance matrix ─────────────────────────────────────────────────
        if len(results) >= 2:
            section_header("Dominance Matrix")
            show(_dominance_chart(results, PLOT_KW))
            st.caption(
                "Each cell = probability that the **row strategy** achieves higher "
                "terminal P&L than the **column strategy** across all simulated paths.  "
                "Diagonal = 50% by definition."
            )

        # ── Single trajectory explorer ───────────────────────────────────────
        section_header("Path Explorer")
        first_res = next(iter(results.values()))
        traj = st.slider("Trajectory #", 0, first_res.pnl.shape[0] - 1,
                         first_res.pnl.shape[0] // 2)

        fig_traj = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                 subplot_titles=["Price", "Inventory", "MtM P&L"],
                                 vertical_spacing=0.05)
        for strat, res in results.items():
            c = strategy_color(strat)
            label = STRATEGIES.get(strat, strat)
            fig_traj.add_trace(go.Scatter(
                x=res.times, y=res.price[traj],
                showlegend=False, line=dict(color="#444c57", width=0.8)),
                row=1, col=1)
            fig_traj.add_trace(go.Scatter(
                x=res.times, y=res.inventory[traj],
                name=label, line=dict(color=c, width=1.5)),
                row=2, col=1)
            fig_traj.add_trace(go.Scatter(
                x=res.times, y=res.mtm[traj],
                showlegend=False, line=dict(color=c, width=1.5)),
                row=3, col=1)

        fig_traj.update_layout(**PLOT_KW, height=500,
                               legend=dict(orientation="h", yanchor="bottom",
                                           y=1.01, xanchor="right", x=1,
                                           font=dict(size=10)))
        show(fig_traj)
