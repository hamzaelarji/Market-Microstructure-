"""Page 8 — RL vs Analytical: Can a learned agent beat the ODE?

Trains tabular Q-learning to control γ(t,n) adaptively.
Benchmarks in Poisson world (RL ≈ ODE) and Hawkes world (RL > ODE).
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from extensions.rl_agent import train_agent, benchmark_agent, QLearningAgent
from ui.styles import PALETTE, PLOT_KW, show, hero_banner, section_header
from ui.components import param_sidebar, metrics_row, insight_box

DEFAULT_GAMMA = 0.01


def render(PARAMS, META, HAS_REAL_DATA):
    hero_banner("🤖", "RL vs Analytical",
                "Can a learned agent match — or beat — the ODE-optimal policy?")

    with st.sidebar:
        section_header("PARAMETERS")
        cfg = param_sidebar(PARAMS, key_prefix="p8", show_model=False)
        st.divider()
        section_header("RL CONFIG")
        n_episodes = st.slider("Training episodes", 100, 2000, 500, 100, key="p8_ep")
        n_test = st.slider("Test episodes", 100, 1000, 300, 50, key="p8_test")
        gamma_levels_str = st.text_input("γ levels (comma-separated)",
                                          value="0.002, 0.01, 0.05", key="p8_gl")

    params = cfg["params"]
    gamma = cfg["gamma"]
    T_val = cfg["T"]

    try:
        gamma_levels = [float(x.strip()) for x in gamma_levels_str.split(",")]
    except ValueError:
        gamma_levels = [0.002, 0.01, 0.05]

    # ── Theory ────────────────────────────────────────────────
    with st.expander("📖 Architecture", expanded=False):
        st.markdown("""
**Agent design** (following Falces Marin et al. 2022, PLOS ONE):

| Component | Choice |
|-----------|--------|
| **State** | (inventory bucket, time bucket) — discretised |
| **Action** | γ level ∈ {γ_low, γ_mid, γ_high} |
| **Quotes** | Guéant closed-form with chosen γ |
| **Reward** | ΔMtM − λ · |inventory| · σ√dt |
| **Algorithm** | Tabular Q-learning (no neural network) |

The RL agent doesn't set quotes directly — it controls **risk aversion** γ(t,n),
then the analytical formulas compute the optimal quotes for that γ.
This hybrid approach inherits the structural guarantees of the Guéant framework
while allowing adaptive risk management.
        """)

    # ── Training ──────────────────────────────────────────────
    tab_poisson, tab_hawkes = st.tabs(["🎲 Poisson World", "⚡ Hawkes World"])

    for tab, fill_mode, tab_name in [
        (tab_poisson, "poisson", "Poisson"),
        (tab_hawkes, "hawkes", "Hawkes"),
    ]:
        with tab:
            hawkes_cfg = {"beta": 10.0, "alpha_self": 2.0, "alpha_cross": 0.5} if fill_mode == "hawkes" else None

            if st.button(f"▶ Train & Benchmark ({tab_name})", type="primary",
                         key=f"p8_run_{fill_mode}"):

                # ── Training ──
                with st.spinner(f"Training Q-learning agent ({n_episodes} episodes, {tab_name} fills)..."):
                    agent, rewards = train_agent(
                        params, n_episodes=n_episodes, T=T_val, N_t=360,
                        fill_mode=fill_mode, hawkes_cfg=hawkes_cfg,
                        gamma_levels=gamma_levels, seed=42)

                section_header("TRAINING CURVE")
                # Smooth rewards
                window = max(10, n_episodes // 20)
                smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")

                fig_train = go.Figure()
                fig_train.add_trace(go.Scatter(
                    y=rewards, mode="lines", name="Episode reward",
                    line=dict(color=PALETTE[6], width=0.5), opacity=0.4))
                fig_train.add_trace(go.Scatter(
                    x=np.arange(window - 1, len(rewards)),
                    y=smooth, mode="lines", name=f"Moving avg ({window})",
                    line=dict(color=PALETTE[0], width=2)))
                fig_train.update_layout(**PLOT_KW, height=300,
                                        xaxis_title="Episode", yaxis_title="Total reward")
                show(fig_train)

                # ── Learned policy heatmap ──
                section_header("LEARNED POLICY: γ(inventory, time)")
                gamma_map = agent.get_gamma_map()
                Q_max = int(params["Q"])
                inv_labels = [str(n) for n in range(-Q_max, Q_max + 1)]
                time_labels = [f"{i}" for i in range(agent.n_time)]

                fig_policy = go.Figure(data=go.Heatmap(
                    z=gamma_map, x=time_labels, y=inv_labels,
                    colorscale="Viridis", colorbar=dict(title="γ"),
                    text=np.round(gamma_map, 4), texttemplate="%{text}"))
                fig_policy.update_layout(**PLOT_KW, height=350,
                                         xaxis_title="Time bucket",
                                         yaxis_title="Inventory (lots)")
                show(fig_policy)

                st.caption("The RL agent learns to vary γ with state — "
                           "higher γ (more risk averse) at extreme inventories, "
                           "lower γ when inventory is near zero.")

                # ── Benchmark ──
                section_header(f"BENCHMARK ({tab_name.upper()} WORLD)")
                with st.spinner(f"Benchmarking ({n_test} test episodes)..."):
                    # Set agent to greedy mode
                    agent.epsilon = 0.0
                    results = benchmark_agent(
                        agent, params, gamma, N_test=n_test, T=T_val, N_t=360,
                        fill_mode=fill_mode, hawkes_cfg=hawkes_cfg, seed=123)

                # PnL distributions
                fig_bench = go.Figure()
                colors = {"RL Agent": PALETTE[3], "ODE Optimal": PALETTE[0], "Naive": PALETTE[1]}
                for name, data in results.items():
                    fig_bench.add_trace(go.Histogram(
                        x=data["pnl"], name=name, opacity=0.6,
                        marker_color=colors.get(name, PALETTE[6]), nbinsx=40))

                fig_bench.update_layout(**PLOT_KW, height=380, barmode="overlay",
                                        xaxis_title="Terminal PnL ($)", yaxis_title="Count")
                show(fig_bench)

                # Metrics
                metrics = {}
                for name, data in results.items():
                    metrics[f"{name} Sharpe"] = data["sharpe"]
                metrics_row(metrics)

                # Summary table
                rows = []
                for name, data in results.items():
                    rows.append({
                        "Strategy": name,
                        "E[PnL]": f"{data['mean_pnl']:.2f}",
                        "Std[PnL]": f"{data['std_pnl']:.2f}",
                        "Sharpe": f"{data['sharpe']:.3f}",
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)

                if fill_mode == "poisson":
                    insight_box("In the Poisson world, RL should approximately match the ODE solution — "
                                "both converge to the same optimal policy. "
                                "If RL significantly underperforms, increase training episodes. "
                                "If it matches, this validates both approaches.")
                else:
                    insight_box("In the Hawkes world, RL can outperform the Guéant policy because "
                                "it learns to adapt to fill clustering. "
                                "The ODE-optimal policy ignores excitation states (y_bid, y_ask) "
                                "and is therefore suboptimal. This is exactly where analytics fail "
                                "and machine learning adds value.")
            else:
                st.info(f"Press **▶ Train & Benchmark ({tab_name})** to run. "
                        f"Training: {n_episodes} episodes (~{n_episodes * 0.01:.0f}s). "
                        f"Testing: {n_test} episodes.")
