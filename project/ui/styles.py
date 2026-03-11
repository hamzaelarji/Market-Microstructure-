"""ui/styles.py — Design system: colors, CSS, Plotly template, helper widgets."""

import streamlit as st

# ─── Color Palette ────────────────────────────────────────────────────────────
TEAL   = "#00d4aa"
RED    = "#ff6b6b"
YELLOW = "#ffd93d"
PURPLE = "#c678dd"
BLUE   = "#61afef"
ORANGE = "#d19a66"
GRAY   = "#abb2bf"

PALETTE = [TEAL, RED, YELLOW, PURPLE, BLUE, ORANGE, GRAY]

# ─── Plotly config ────────────────────────────────────────────────────────────
PLOT_KW = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,17,23,0.6)",
    font=dict(size=11, color="#c8cdd5"),
    margin=dict(l=50, r=20, t=44, b=36),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
_CSS = """
<style>

/* ── Base & layout ── */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 0.8rem; padding-bottom: 1rem; max-width: 1400px; }
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #12151c;
    border-right: 1px solid #1e2330;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
.sidebar-logo {
    font-size: 1.4rem; font-weight: 700; color: #00d4aa;
    letter-spacing: -0.5px; padding: 0.5rem 0 1rem 0;
}
.sidebar-cite {
    font-size: 0.7rem; color: #555e6e; line-height: 1.4;
    border-top: 1px solid #1e2330; padding-top: 0.8rem; margin-top: 1rem;
}

/* ── Tabs (top-level) ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #1e2330;
    background: transparent;
    padding: 0 0.2rem;
}
.stTabs [data-baseweb="tab"] {
    height: 36px;
    border-radius: 6px 6px 0 0;
    padding: 0 14px;
    font-size: 0.78rem;
    font-weight: 500;
    color: #7a8290;
    background: transparent;
    border: none;
    transition: color 0.15s, background 0.15s;
}
.stTabs [aria-selected="true"] {
    color: #00d4aa !important;
    background: rgba(0,212,170,0.06) !important;
    border-bottom: 2px solid #00d4aa !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #c8cdd5; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #12151c;
    border: 1px solid #1e2330;
    border-radius: 10px;
    padding: 14px 18px;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: #2a3040; }
[data-testid="stMetricLabel"] { font-size: 0.72rem; opacity: 0.5; letter-spacing: 0.5px; }
[data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 600; }
[data-testid="stMetricDelta"] { font-size: 0.75rem; }

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00d4aa, #00a882);
    color: #0e1117;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.15s, transform 0.1s;
}
.stButton > button[kind="primary"]:hover { opacity: 0.9; transform: translateY(-1px); }
.stButton > button[kind="primary"]:active { transform: translateY(0); }

/* ── Info / warning boxes ── */
.stAlert { border-radius: 8px; border-left-width: 3px; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0e1724 0%, #111823 100%);
    border: 1px solid #1e2d3e;
    border-left: 3px solid #00d4aa;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.hero-icon { font-size: 1.8rem; }
.hero-title {
    font-size: 1.15rem; font-weight: 700;
    background: linear-gradient(90deg, #00d4aa, #61afef);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-subtitle { font-size: 0.78rem; color: #5a6270; margin: 2px 0 0 0; }

/* ── Section header ── */
.section-header {
    font-size: 0.8rem; font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; color: #4a5260;
    border-bottom: 1px solid #1e2330;
    padding-bottom: 6px; margin: 18px 0 12px 0;
}

/* ── Dividers ── */
hr { border-color: #1e2330; opacity: 1; }

/* ── Tables / dataframes ── */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* ── Progress bars ── */
.stProgress > div > div { background: linear-gradient(90deg, #00d4aa, #61afef); }

/* ── Selectbox / inputs ── */
.stSelectbox > div, .stNumberInput > div > div > input {
    background: #12151c;
    border-color: #1e2330;
}

/* ── Pnl glow ── */
.pnl-positive { color: #00d4aa; text-shadow: 0 0 12px rgba(0,212,170,0.4); }
.pnl-negative { color: #ff6b6b; text-shadow: 0 0 12px rgba(255,107,107,0.4); }

</style>
"""


def apply_styles():
    """Inject CSS into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def show(fig, **kw):
    """Render a Plotly figure with dark bg and full width."""
    st.plotly_chart(fig, use_container_width=True, theme=None, **kw)


def hero_banner(icon: str, title: str, subtitle: str = ""):
    """Render a top-of-tab hero banner."""
    st.markdown(
        f"""<div class="hero-banner">
          <div class="hero-icon">{icon}</div>
          <div>
            <p class="hero-title">{title}</p>
            <p class="hero-subtitle">{subtitle}</p>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


def section_header(text: str):
    """Render a small section header with bottom border."""
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def pnl_number(value: float, prefix: str = "$") -> str:
    """Format a P&L value with HTML color class."""
    cls = "pnl-positive" if value >= 0 else "pnl-negative"
    sign = "+" if value >= 0 else ""
    return f'<span class="{cls}">{prefix}{sign}{value:,.2f}</span>'


def strategy_color(name: str) -> str:
    """Map strategy name to a consistent color."""
    _MAP = {
        "optimal": TEAL,
        "naive": RED,
        "closed_form": PURPLE,
        "inv_skew_only": ORANGE,
    }
    return _MAP.get(name, GRAY)
