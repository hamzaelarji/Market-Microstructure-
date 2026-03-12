"""ui/components.py — Shared helpers for parameter inputs and display."""

import numpy as np
import streamlit as st
from market_making.params.assets import IG, HY

# ─── Presets ──────────────────────────────────────────────────
PRESETS = {
    "CDX.NA.IG (paper)": IG,
    "CDX.NA.HY (paper)": HY,
}


def add_calibrated_presets(PARAMS):
    """Merge calibrated crypto presets into the global preset dict."""
    merged = dict(PRESETS)
    for sym, p in PARAMS.items():
        merged[sym] = p
    return merged


def param_row(key, presets=None, default_gamma=0.01, show_model=False,
              show_T=False, n_cols=6):
    """Render a compact horizontal parameter row.  Returns (params, gamma, xi, T)."""
    if presets is None:
        presets = PRESETS

    cols = st.columns(n_cols)

    with cols[0]:
        preset_name = st.selectbox("Preset", list(presets.keys()) + ["Custom"],
                                   key=f"{key}_preset")
    d = presets.get(preset_name, {"sigma": 5.0, "A": 5.0, "k": 3.0,
                                   "Delta": 100.0, "Q": 4})

    with cols[1]:
        sigma = st.number_input("σ", value=d["sigma"], format="%.6g",
                                step=d["sigma"] * 0.1, key=f"{key}_s")
    with cols[2]:
        A = st.number_input("A", value=d["A"], format="%.6g",
                            step=d["A"] * 0.1, key=f"{key}_A")
    with cols[3]:
        k = st.number_input("k", value=d["k"], format="%.6g",
                            step=d["k"] * 0.1, key=f"{key}_k")
    with cols[4]:
        Delta = st.number_input("Δ", value=d["Delta"], format="%.4g",
                                step=d["Delta"] * 0.1, key=f"{key}_D")
    with cols[5]:
        Q = st.number_input("Q", value=int(d.get("Q", 4)), min_value=1,
                            max_value=10, step=1, key=f"{key}_Q")

    # Second row for gamma, model, T
    cols2 = st.columns(n_cols)
    with cols2[0]:
        gamma = st.number_input("γ", value=default_gamma, format="%.6g",
                                step=0.001, key=f"{key}_g")
    xi = gamma
    model = "A"
    if show_model:
        with cols2[1]:
            ml = st.radio("Model", ["A (ξ=γ)", "B (ξ=0)"], horizontal=True,
                          key=f"{key}_mdl")
            if "B" in ml:
                xi = 0.0
                model = "B"

    T_val = 3600.0
    if show_T:
        with cols2[2]:
            T_val = st.number_input("T (s)", value=3600.0, step=600.0,
                                    key=f"{key}_T")

    params = {"sigma": sigma, "A": A, "k": k, "Delta": Delta, "Q": int(Q)}
    return params, gamma, xi, T_val


def insight(text, icon="💡"):
    st.info(f"{icon} {text}")
