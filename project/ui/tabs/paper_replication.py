"""Page 1 — Paper Replication.

Interactive reproduction of Guéant (2017) — each section is an
st.fragment so parameter changes only re-render that section.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_making.core.solver_1d import solve_general
from market_making.core.solver_2d import solve_2d
from market_making.core.closed_form import approx_quotes
from market_making.core.intensity import C_coeff, fill_prob
from market_making.core.hawkes import HawkesState, DEFAULT_HAWKES_CFG, softplus
from market_making.simulation.backtest import compare_strategies
from market_making.params.assets import IG, HY, GAMMA as PAPER_GAMMA, RHO

from ui.styles import PALETTE, PLOT_KW, show, hero_banner, section_header
from ui.components import param_row, add_calibrated_presets, insight, PRESETS
from ui.loaders import load_calibrated_params, load_mid_prices

# ─── Load calibrated data once ────────────────────────────────
_PARAMS, _META, _HAS_REAL = load_calibrated_params()
_ALL_PRESETS = add_calibrated_presets(_PARAMS)

# Paper-specific presets (with paper gamma)
_PAPER_PRESETS = {
    "CDX.NA.IG (paper)": IG,
    "CDX.NA.HY (paper)": HY,
}


# ═══════════════════════════════════════════════════════════════
#  §1  SINGLE-ASSET OPTIMAL QUOTES
# ═══════════════════════════════════════════════════════════════

@st.fragment
def section_single_asset():
    section_header("§1 — SINGLE-ASSET OPTIMAL QUOTES")
    st.caption("Paper §3 + §6.1–6.2 — Figures 1–5 (IG) and 10–14 (HY). "
               "Solve the ODE (Eq. 3.9) and extract δ^bid, δ^ask, spread, skew.")

    params, gamma, xi, T_val = param_row("s1", _ALL_PRESETS,
                                          default_gamma=6e-5, show_model=True, show_T=True)
    Q = int(params["Q"])
    sigma, A, k, Delta = params["sigma"], params["A"], params["k"], params["Delta"]

    try:
        N_t = max(300, int(T_val))
        sol = solve_general(params, gamma, T_val, xi=xi, N_t=N_t)
    except Exception as e:
        st.error(f"ODE solve failed: {e}")
        return

    lots = sol["lots"]
    db = sol["delta_bid"][0, :]
    da = sol["delta_ask"][0, :]

    # ── 2×2 quote panel ──
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.10,
                        subplot_titles=["δ^bid(0, n)", "δ^ask(0, n)",
                                        "Spread(0, n)", "Skew(0, n)"])

    def _trace(row, col, y, color, name, legend=False):
        m = np.isfinite(y)
        fig.add_trace(go.Scatter(
            x=lots[m], y=y[m], mode="lines+markers", name=name,
            marker=dict(size=6, color=color), line=dict(color=color, width=2),
            showlegend=legend), row=row, col=col)

    _trace(1, 1, db, PALETTE[0], "bid", True)
    _trace(1, 2, da, PALETTE[1], "ask", True)
    _trace(2, 1, db + da, PALETTE[4], "spread", True)
    _trace(2, 2, db - da, PALETTE[3], "skew", True)

    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(title_text="n (lots)", row=r, col=c)
    fig.update_layout(**PLOT_KW, height=500,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    show(fig)

    # ── Metrics ──
    xi_D = xi * Delta
    if abs(xi_D) > 1e-12:
        d_s = (1.0 / xi_D) * np.log(1 + xi_D / k)
    else:
        d_s = 1.0 / k
    C = C_coeff(xi_D, k)
    omega = np.sqrt(gamma * sigma ** 2 / (2.0 * A * Delta * k * C)) if A * Delta * k * C > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("δ_static", f"{d_s:.4g}")
    m2.metric("ω (slope)", f"{omega:.4g}")
    m3.metric("ξΔ", f"{xi_D:.4g}")
    m4.metric("Spread(0,0)", f"{float(db[Q] + da[Q]):.4g}" if np.isfinite(db[Q]) else "—")

    # ── θ value function ──
    show_theta = st.checkbox("Show θ(0,n) value function", value=False, key="s1_theta")
    if show_theta:
        theta_0 = sol["theta"][0, :]
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=lots, y=theta_0, mode="lines+markers",
            marker=dict(size=7, color=PALETTE[3]), line=dict(color=PALETTE[3], width=2)))
        fig_t.update_layout(**PLOT_KW, height=300,
                            xaxis_title="n (lots)", yaxis_title="θ(0, n)")
        show(fig_t)
        st.caption("θ is concave and symmetric — its discrete gradient determines the quotes.")

    # ── 3D surface ──
    show_3d = st.checkbox("Show 3D surface δ^bid(t, n)", value=False, key="s1_3d")
    if show_3d:
        step_t = max(1, len(sol["times"]) // 80)
        t_sub = sol["times"][::step_t]
        db_sub = np.where(np.isfinite(sol["delta_bid"][::step_t, :]),
                          sol["delta_bid"][::step_t, :], np.nan)
        fig3d = go.Figure(data=[go.Surface(z=db_sub, x=lots, y=t_sub,
                          colorscale="Viridis", colorbar=dict(title="δ^bid"))])
        fig3d.update_layout(**PLOT_KW, height=450,
                            scene=dict(xaxis_title="n", yaxis_title="t (s)",
                                       zaxis_title="δ^bid", bgcolor="rgba(0,0,0,0)"))
        show(fig3d)

    insight("The optimal MM skews quotes to mean-revert inventory: "
            "long → widen bid, short → tighten ask. Skew(0,0) = 0 by symmetry.")


# ═══════════════════════════════════════════════════════════════
#  §2  CLOSED-FORM APPROXIMATION
# ═══════════════════════════════════════════════════════════════

@st.fragment
def section_closed_form():
    section_header("§2 — CLOSED-FORM APPROXIMATION")
    st.caption("Paper §4 — Guéant–Lehalle–Fernandez-Tapia formulas vs exact ODE. "
               "Figures 6–9 (σ and σ/2 overlays).")

    c_pre, c_mult = st.columns([4, 2])
    with c_pre:
        params, gamma, xi, T_val = param_row("s2", _ALL_PRESETS,
                                              default_gamma=6e-5, show_model=True, show_T=True)
    with c_mult:
        sigma_mult = st.select_slider("σ multiplier",
                                       options=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
                                       value=1.0, key="s2_smult")

    params_eff = params.copy()
    params_eff["sigma"] = params["sigma"] * sigma_mult
    Q = int(params["Q"])

    try:
        N_t = max(300, int(T_val))
        sol = solve_general(params_eff, gamma, T_val, xi=xi, N_t=N_t)
        n_arr = np.arange(-Q + 1, Q)
        db_cf, da_cf = approx_quotes(n_arr, params_eff, gamma, xi=xi)
    except Exception as e:
        st.error(f"Solve failed: {e}")
        return

    lots = sol["lots"]
    db_ode = sol["delta_bid"][0, :]
    da_ode = sol["delta_ask"][0, :]

    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.10,
                        subplot_titles=["δ^bid(0, n)", "δ^ask(0, n)",
                                        "Spread(0, n)", "Skew(0, n)"])

    def _pair(row, col, y_ode, y_cf, legend=False):
        m = np.isfinite(y_ode)
        fig.add_trace(go.Scatter(x=lots[m], y=y_ode[m], mode="markers", name="ODE",
            marker=dict(size=7, color=PALETTE[0], symbol="x"), showlegend=legend),
            row=row, col=col)
        fig.add_trace(go.Scatter(x=n_arr, y=y_cf, mode="lines", name="Closed-form",
            line=dict(dash="dash", color=PALETTE[1], width=2), showlegend=legend),
            row=row, col=col)

    _pair(1, 1, db_ode, db_cf, True)
    _pair(1, 2, da_ode, da_cf)
    _pair(2, 1, db_ode + da_ode, db_cf + da_cf)
    _pair(2, 2, db_ode - da_ode, db_cf - da_cf)

    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(title_text="n", row=r, col=c)
    fig.update_layout(**PLOT_KW, height=500,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    show(fig)

    # Error metrics
    db_ode_sub = sol["delta_bid"][0, 1:-1]
    if len(db_cf) == len(db_ode_sub):
        max_err = float(np.nanmax(np.abs(db_ode_sub - db_cf)))
        spread_cf = db_cf + da_cf
        mean_sp = float(np.nanmean(spread_cf)) if np.nanmean(spread_cf) != 0 else 1
        rel_err = float(np.nanmax(np.abs((db_ode_sub + sol["delta_ask"][0, 1:-1]) - spread_cf)) / abs(mean_sp))
    else:
        max_err, rel_err = 0, 0

    e1, e2, e3 = st.columns(3)
    e1.metric("max |δ_ODE − δ_CF|", f"{max_err:.4g}")
    e2.metric("Rel. spread error", f"{rel_err:.2%}")
    e3.metric("σ effective", f"{params_eff['sigma']:.4g}")

    insight("The CF approximation works well when ωΔ is small. "
            "Halving σ dramatically improves accuracy (try σ × 0.5).")


# ═══════════════════════════════════════════════════════════════
#  §3  MODEL A vs MODEL B
# ═══════════════════════════════════════════════════════════════

@st.fragment
def section_model_comparison():
    section_header("§3 — MODEL A vs MODEL B")
    st.caption("Paper §3.4 — ξ = γ (Model A, CARA utility) vs ξ = 0 (Model B, mean-variance). "
               "Same ODE, different Hamiltonian.")

    params, gamma, _, T_val = param_row("s3", _ALL_PRESETS,
                                         default_gamma=6e-5, show_T=True)
    Q = int(params["Q"])

    try:
        N_t = max(300, int(T_val))
        sol_a = solve_general(params, gamma, T_val, xi=gamma, N_t=N_t)
        sol_b = solve_general(params, gamma, T_val, xi=0.0, N_t=N_t)
    except Exception as e:
        st.error(f"Solve failed: {e}")
        return

    lots = sol_a["lots"]
    spread_a = sol_a["delta_bid"][0, :] + sol_a["delta_ask"][0, :]
    spread_b = sol_b["delta_bid"][0, :] + sol_b["delta_ask"][0, :]
    skew_a = sol_a["delta_bid"][0, :] - sol_a["delta_ask"][0, :]
    skew_b = sol_b["delta_bid"][0, :] - sol_b["delta_ask"][0, :]

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                        subplot_titles=["Spread(0, n)", "Skew(0, n)"])

    for y, name, color in [(spread_a, "A (ξ=γ)", PALETTE[0]), (spread_b, "B (ξ=0)", PALETTE[1])]:
        m = np.isfinite(y)
        fig.add_trace(go.Scatter(x=lots[m], y=y[m], mode="lines+markers", name=name,
            marker=dict(size=6, color=color), line=dict(color=color, width=2)), row=1, col=1)
    for y, name, color in [(skew_a, "A (ξ=γ)", PALETTE[0]), (skew_b, "B (ξ=0)", PALETTE[1])]:
        m = np.isfinite(y)
        fig.add_trace(go.Scatter(x=lots[m], y=y[m], mode="lines+markers", name=name,
            marker=dict(size=6, color=color), line=dict(color=color, width=2),
            showlegend=False), row=1, col=2)

    fig.update_xaxes(title_text="n (lots)", row=1, col=1)
    fig.update_xaxes(title_text="n (lots)", row=1, col=2)
    fig.update_layout(**PLOT_KW, height=370,
                      legend=dict(orientation="h", yanchor="bottom", y=1.05))
    show(fig)

    # Comparative statics
    k = params["k"]
    Delta = params["Delta"]
    d_s_a = (1.0 / (gamma * Delta)) * np.log(1 + gamma * Delta / k) if abs(gamma * Delta) > 1e-12 else 1.0 / k
    d_s_b = 1.0 / k
    C_a = C_coeff(gamma * Delta, k)
    C_b = C_coeff(0, k)
    s = params["sigma"]
    omega_a = np.sqrt(gamma * s**2 / (2.0 * params["A"] * Delta * k * C_a)) if params["A"] * Delta * k * C_a > 0 else 0
    omega_b = np.sqrt(gamma * s**2 / (2.0 * params["A"] * Delta * k * C_b)) if params["A"] * Delta * k * C_b > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("δ_static A", f"{d_s_a:.4g}")
    c2.metric("δ_static B", f"{d_s_b:.4g}")
    c3.metric("ω A", f"{omega_a:.4g}")
    c4.metric("ω B", f"{omega_b:.4g}")

    insight("Model A (ξ=γ) penalises non-execution risk on top of price risk → "
            "tighter spread, more aggressive skewing. Model B (ξ=0) only penalises inventory.")


# ═══════════════════════════════════════════════════════════════
#  §4  MULTI-ASSET
# ═══════════════════════════════════════════════════════════════

@st.fragment
def section_multi_asset():
    section_header("§4 — MULTI-ASSET MARKET MAKING")
    st.caption("Paper §5 + §6.3 — Figures 17–19. Two correlated assets (IG + HY). "
               "Optimal bid for each depends on inventory in *both* assets.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rho = st.slider("ρ", 0.0, 1.0, 0.9, 0.05, key="s4_rho")
    with c2:
        gamma_2d = st.number_input("γ", value=PAPER_GAMMA, format="%.6g", key="s4_g")
    with c3:
        T_2d = st.number_input("T (s)", value=3600.0, step=600.0, key="s4_T")
    with c4:
        N_t_2d = st.slider("N_t", 100, 1000, 300, 50, key="s4_Nt")

    try:
        with st.spinner("Solving 2D ODE..."):
            sol2d = solve_2d(IG, HY, gamma_2d, rho, T_2d, xi=gamma_2d, N_t=N_t_2d)
    except Exception as e:
        st.error(f"2D solver failed: {e}")
        return

    grid = sol2d["grid"]
    idx = sol2d["idx"]
    Q1, Q2 = int(IG["Q"]), int(HY["Q"])
    lots1 = np.arange(-Q1, Q1 + 1)
    lots2 = np.arange(-Q2, Q2 + 1)

    # Build surface matrices
    db1_mat = np.full((len(lots1), len(lots2)), np.nan)
    db2_mat = np.full((len(lots1), len(lots2)), np.nan)
    for i, n1 in enumerate(lots1):
        for j, n2 in enumerate(lots2):
            if (n1, n2) in idx:
                db1_mat[i, j] = sol2d["delta_bid_1"][0, idx[(n1, n2)]]
                db2_mat[i, j] = sol2d["delta_bid_2"][0, idx[(n1, n2)]]

    col_a, col_b = st.columns(2)
    with col_a:
        fig17 = go.Figure(data=[go.Surface(z=db1_mat, x=lots2, y=lots1,
                          colorscale="Viridis", colorbar=dict(title="δ^bid₁"))])
        fig17.update_layout(**PLOT_KW, height=420, title="δ^bid IG (Fig 17)",
                            scene=dict(xaxis_title="n₂ (HY)", yaxis_title="n₁ (IG)",
                                       zaxis_title="δ^bid₁"))
        show(fig17)

    with col_b:
        fig18 = go.Figure(data=[go.Surface(z=db2_mat, x=lots2, y=lots1,
                          colorscale="Plasma", colorbar=dict(title="δ^bid₂"))])
        fig18.update_layout(**PLOT_KW, height=420, title="δ^bid HY (Fig 18)",
                            scene=dict(xaxis_title="n₂ (HY)", yaxis_title="n₁ (IG)",
                                       zaxis_title="δ^bid₂"))
        show(fig18)

    # Cross-section (Fig 19): fix n2=0, sweep rho
    show_cross = st.checkbox("Show cross-asset effect vs ρ (Fig 19)", value=False, key="s4_cross")
    if show_cross:
        rho_vals = [0.0, 0.3, 0.6, 0.9]
        fig19 = go.Figure()
        colors_rho = [PALETTE[0], PALETTE[4], PALETTE[5], PALETTE[1]]
        for rv, clr in zip(rho_vals, colors_rho):
            try:
                sol_rho = solve_2d(IG, HY, gamma_2d, rv, T_2d, xi=gamma_2d,
                                   N_t=min(N_t_2d, 200))
                y_vals = [sol_rho["delta_bid_1"][0, sol_rho["idx"][(n1, 0)]]
                          if (n1, 0) in sol_rho["idx"] else np.nan for n1 in lots1]
                fig19.add_trace(go.Scatter(x=lots1, y=y_vals, mode="lines+markers",
                    name=f"ρ={rv}", line=dict(color=clr, width=2), marker=dict(size=5)))
            except Exception:
                pass
        fig19.update_layout(**PLOT_KW, height=350,
                            xaxis_title="n₁ (IG lots)", yaxis_title="δ^bid₁ at n₂=0")
        show(fig19)

    insight("Higher ρ → stronger cross-asset effect: inventory in one asset shifts quotes in the other. "
            "At ρ=0, the 2D problem decouples into two independent 1D problems.")


# ═══════════════════════════════════════════════════════════════
#  §5  MONTE CARLO VALIDATION
# ═══════════════════════════════════════════════════════════════

@st.fragment
def section_monte_carlo():
    section_header("§5 — MONTE CARLO VALIDATION")
    st.caption("Extension — does the ODE-optimal strategy actually generate profit? "
               "Compare optimal vs naive via Monte Carlo simulation.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mc_preset = st.selectbox("Preset", list(_ALL_PRESETS.keys()), key="s5_pre")
    mc_params = _ALL_PRESETS[mc_preset]
    with c2:
        mc_gamma = st.number_input("γ", value=0.01, format="%.4g", key="s5_g")
    with c3:
        mc_nsim = st.slider("N_sim", 100, 3000, 500, 100, key="s5_nsim")
    with c4:
        mc_run = st.button("▶ Run", type="primary", key="s5_run")

    if mc_run:
        with st.spinner("Running optimal + naive + CF backtests..."):
            try:
                results = compare_strategies(mc_params, gamma=mc_gamma, T=3600.0,
                                             N_sim=mc_nsim, seed=42,
                                             strategies=["optimal", "naive", "closed_form"])
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                return

        # PnL histogram
        fig_pnl = go.Figure()
        colors = {"optimal": PALETTE[0], "naive": PALETTE[1], "closed_form": PALETTE[3]}
        for name, r in results.items():
            fig_pnl.add_trace(go.Histogram(x=r.pnl, name=name.replace("_", " ").title(),
                opacity=0.6, marker_color=colors.get(name, PALETTE[6]), nbinsx=50))
        fig_pnl.update_layout(**PLOT_KW, height=350, barmode="overlay",
                              xaxis_title="Terminal PnL ($)", yaxis_title="Count")
        show(fig_pnl)

        # Trajectory + inventory
        col_t, col_i = st.columns(2)
        res_opt = results["optimal"]
        res_naive = results["naive"]
        times = res_opt.times

        with col_t:
            fig_tr = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                   subplot_titles=["Mid Price", "MtM ($)"])
            fig_tr.add_trace(go.Scatter(x=times, y=res_opt.price[0], mode="lines",
                line=dict(color=PALETTE[6], width=1), showlegend=False), row=1, col=1)
            fig_tr.add_trace(go.Scatter(x=times, y=res_opt.mtm[0], mode="lines",
                name="Optimal", line=dict(color=PALETTE[0], width=1.5)), row=2, col=1)
            fig_tr.add_trace(go.Scatter(x=times, y=res_naive.mtm[0], mode="lines",
                name="Naive", line=dict(color=PALETTE[1], width=1.5)), row=2, col=1)
            fig_tr.update_layout(**PLOT_KW, height=350)
            show(fig_tr)

        with col_i:
            fig_inv = go.Figure()
            fig_inv.add_trace(go.Scatter(x=times, y=np.mean(np.abs(res_opt.inventory), axis=0),
                mode="lines", name="Optimal E[|n|]", line=dict(color=PALETTE[0], width=2)))
            fig_inv.add_trace(go.Scatter(x=times, y=np.mean(np.abs(res_naive.inventory), axis=0),
                mode="lines", name="Naive E[|n|]", line=dict(color=PALETTE[1], width=2)))
            fig_inv.update_layout(**PLOT_KW, height=350,
                                  xaxis_title="t (s)", yaxis_title="E[|inventory|]")
            show(fig_inv)

        # Summary table
        rows = []
        for name, r in results.items():
            s = r.summary()
            rows.append({"Strategy": name, "E[PnL]": f"{s['mean_pnl']:.2f}",
                         "Std": f"{s['std_pnl']:.2f}", "Sharpe": f"{s['sharpe']:.3f}",
                         "CE": f"{s['CE']:.2f}", "Fills": f"{s['mean_fills']:.0f}"})
        st.dataframe(rows, use_container_width=True, hide_index=True)

        insight("Optimal has LOWER mean PnL but HIGHER Sharpe and CE. "
                "It maximises CARA utility, not expected wealth.")


# ═══════════════════════════════════════════════════════════════
#  §6  HAWKES vs POISSON
# ═══════════════════════════════════════════════════════════════

@st.fragment
def section_hawkes():
    section_header("§6 — HAWKES vs POISSON FILL DYNAMICS")
    st.caption("Extension — what happens when fills cluster? Hawkes self-excitation "
               "adds state variables (y_bid, y_ask), breaking the ODE's tractability.")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        h_preset = st.selectbox("Preset", list(_ALL_PRESETS.keys()), key="s6_pre")
    h_params = _ALL_PRESETS[h_preset]
    with c2:
        h_gamma = st.number_input("γ", value=0.01, format="%.4g", key="s6_g")
    with c3:
        alpha_self = st.slider("α_self", 0.0, 5.0, 2.0, 0.5, key="s6_as")
    with c4:
        alpha_cross = st.slider("α_cross", 0.0, 3.0, 0.5, 0.25, key="s6_ac")
    with c5:
        beta = st.slider("β", 1.0, 30.0, 10.0, 1.0, key="s6_beta")

    hawkes_cfg = {"beta": beta, "alpha_self": alpha_self, "alpha_cross": alpha_cross}

    n_sim_h = st.slider("N_sim", 100, 1000, 300, 50, key="s6_nsim")
    run_h = st.button("▶ Run comparison", type="primary", key="s6_run")

    if run_h:
        sigma, A, k = h_params["sigma"], h_params["A"], h_params["k"]
        Delta, Q = h_params["Delta"], int(h_params["Q"])
        T_h = 3600.0
        N_t_h = max(300, int(T_h))
        dt = T_h / N_t_h

        try:
            sol = solve_general(h_params, h_gamma, T_h, xi=h_gamma, N_t=N_t_h)
        except Exception as e:
            st.error(f"Solve failed: {e}")
            return

        rng = np.random.default_rng(42)
        pnl_p, pnl_h = [], []
        # Store one path for visualisation
        path_lam_p, path_lam_h, path_y_b, path_y_a = None, None, None, None

        for sim_i in range(n_sim_h):
            seed_i = int(rng.integers(1e9))
            for mode in ("poisson", "hawkes"):
                rng_i = np.random.default_rng(seed_i)
                S, X, n = 0.0, 0.0, 0
                hs = HawkesState(hawkes_cfg) if mode == "hawkes" else None

                lam_arr = np.zeros(N_t_h) if sim_i == 0 else None
                y_b_arr = np.zeros(N_t_h) if (sim_i == 0 and mode == "hawkes") else None
                y_a_arr = np.zeros(N_t_h) if (sim_i == 0 and mode == "hawkes") else None

                for t_idx in range(N_t_h):
                    i_lot = n + Q
                    db = sol["delta_bid"][t_idx, i_lot] if (n < Q and np.isfinite(sol["delta_bid"][t_idx, i_lot])) else np.inf
                    da = sol["delta_ask"][t_idx, i_lot] if (n > -Q and np.isfinite(sol["delta_ask"][t_idx, i_lot])) else np.inf

                    S += sigma * np.sqrt(dt) * rng_i.standard_normal()
                    mu_b = A * np.exp(-k * max(db, 0)) if db < np.inf else 0
                    mu_a = A * np.exp(-k * max(da, 0)) if da < np.inf else 0

                    if mode == "hawkes" and hs:
                        lam_b = hs.lambda_bid(mu_b)
                        lam_a = hs.lambda_ask(mu_a)
                        if y_b_arr is not None:
                            y_b_arr[t_idx] = hs.y_bid
                            y_a_arr[t_idx] = hs.y_ask
                    else:
                        lam_b, lam_a = mu_b, mu_a

                    if lam_arr is not None:
                        lam_arr[t_idx] = lam_b

                    bf = rng_i.random() < fill_prob(lam_b, dt) if db < np.inf else False
                    af = rng_i.random() < fill_prob(lam_a, dt) if da < np.inf else False

                    if bf and n < Q:
                        X -= (S - db) * Delta; n += 1
                    if af and n > -Q:
                        X += (S + da) * Delta; n -= 1

                    if mode == "hawkes" and hs:
                        hs.step(dt, bf, af)

                pnl_val = X + n * Delta * S
                if mode == "poisson":
                    pnl_p.append(pnl_val)
                    if sim_i == 0:
                        path_lam_p = lam_arr.copy()
                else:
                    pnl_h.append(pnl_val)
                    if sim_i == 0:
                        path_lam_h = lam_arr.copy() if lam_arr is not None else None
                        path_y_b = y_b_arr.copy() if y_b_arr is not None else None
                        path_y_a = y_a_arr.copy() if y_a_arr is not None else None

        pnl_p, pnl_h = np.array(pnl_p), np.array(pnl_h)
        t_arr = np.linspace(0, T_h, N_t_h, endpoint=False)

        # Intensity traces
        if path_lam_p is not None and path_lam_h is not None:
            fig_lam = make_subplots(rows=1, cols=2, subplot_titles=["Poisson λ_bid(t)", "Hawkes λ_bid(t)"])
            fig_lam.add_trace(go.Scatter(x=t_arr, y=path_lam_p, mode="lines",
                line=dict(color=PALETTE[6], width=1), showlegend=False), row=1, col=1)
            # For hawkes, recompute intensity trace from the first sim path
            fig_lam.add_trace(go.Scatter(x=t_arr, y=path_lam_h if path_lam_h is not None else [],
                mode="lines", line=dict(color=PALETTE[0], width=1), showlegend=False), row=1, col=2)
            fig_lam.update_layout(**PLOT_KW, height=250)
            show(fig_lam)

        # Excitation
        if path_y_b is not None:
            fig_y = go.Figure()
            fig_y.add_trace(go.Scatter(x=t_arr, y=path_y_b, mode="lines", name="y_bid",
                line=dict(color=PALETTE[0], width=1.5)))
            fig_y.add_trace(go.Scatter(x=t_arr, y=path_y_a, mode="lines", name="y_ask",
                line=dict(color=PALETTE[1], width=1.5)))
            fig_y.update_layout(**PLOT_KW, height=220,
                                xaxis_title="t (s)", yaxis_title="Excitation y(t)")
            show(fig_y)

        # PnL comparison
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(x=pnl_p, name="Poisson world", opacity=0.6,
            marker_color=PALETTE[6], nbinsx=40))
        fig_mc.add_trace(go.Histogram(x=pnl_h, name="Hawkes world", opacity=0.6,
            marker_color=PALETTE[0], nbinsx=40))
        fig_mc.update_layout(**PLOT_KW, height=320, barmode="overlay",
                             xaxis_title="Terminal PnL ($)", yaxis_title="Count")
        show(fig_mc)

        sp = float(np.mean(pnl_p) / max(np.std(pnl_p), 1e-12))
        sh = float(np.mean(pnl_h) / max(np.std(pnl_h), 1e-12))
        m1, m2, m3 = st.columns(3)
        m1.metric("Sharpe (Poisson)", f"{sp:.3f}")
        m2.metric("Sharpe (Hawkes)", f"{sh:.3f}")
        m3.metric("Degradation", f"{(1 - sh / max(sp, 1e-12)) * 100:.1f}%")

        insight("Same Guéant policy underperforms in Hawkes world — it ignores fill clustering. "
                "Hawkes adds 2 state variables (y_b, y_a) → 4D state space, breaking analytical tractability.")


# ═══════════════════════════════════════════════════════════════
#  MAIN RENDER
# ═══════════════════════════════════════════════════════════════

def render():
    hero_banner("📘", "Paper Replication",
                "Interactive reproduction of Guéant (2017) — Optimal Market Making")

    section_single_asset()
    st.divider()
    section_closed_form()
    st.divider()
    section_model_comparison()
    st.divider()
    section_multi_asset()
    st.divider()
    section_monte_carlo()
    st.divider()
    section_hawkes()
