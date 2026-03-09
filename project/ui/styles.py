"""ui/styles.py — Visual constants and CSS injection."""

import streamlit as st

PALETTE = ["#00d4aa", "#ff6b6b", "#ffd93d"]

PLOT_KW = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=11),
    margin=dict(l=50, r=20, t=36, b=36),
)

_CSS = """
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
"""


def apply_styles():
    """Inject the minimal polish CSS into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def show(fig, **kw):
    """Render a Plotly figure with native theme disabled so the dark bg works."""
    st.plotly_chart(fig, use_container_width=True, theme="streamlit", **kw)
