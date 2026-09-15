"""
Visual system for the dashboard.

One place for the CSS, the Plotly template and the handful of reusable
components, so every section looks like it belongs to the same product.

Design notes
------------
Palette : deep teal carries "support exists", ochre carries "nobody knows",
          plum carries perception and stigma, rose carries risk. A reader
          learns the colour language once and it holds across 40+ charts.
Type    : Sora for headings and large figures (geometric, slightly editorial),
          Inter for everything else, with tabular figures so numbers in a
          column line up.
Surface : a cool paper canvas with flat white cards and hairline rules rather
          than drop shadows — the data should be the only thing with weight.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src import config as C

ICON = {
    "overview": "◆",
    "quality": "◇",
    "people": "○",
    "workplace": "▣",
    "stigma": "◈",
    "drivers": "▲",
    "model": "◐",
    "map": "⬡",
    "explorer": "▤",
    "insights": "★",
}


# --------------------------------------------------------------------------
# Plotly template
# --------------------------------------------------------------------------

def register_template() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=f"{C.FONT_BODY}, system-ui, sans-serif", size=13,
                  color=C.COLORS["ink_soft"]),
        title=dict(font=dict(family=f"{C.FONT_HEADING}, system-ui, sans-serif", size=16,
                             color=C.COLORS["ink"]), x=0, xanchor="left", pad=dict(b=12)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[C.COLORS["teal"], C.COLORS["plum"], C.COLORS["ochre"], C.COLORS["sky"],
                  C.COLORS["rose"], C.COLORS["slate"]],
        xaxis=dict(gridcolor=C.COLORS["line"], linecolor=C.COLORS["line"], zeroline=False,
                   ticks="outside", tickcolor=C.COLORS["line"],
                   title=dict(font=dict(size=12, color=C.COLORS["slate"]))),
        yaxis=dict(gridcolor=C.COLORS["line"], linecolor=C.COLORS["line"], zeroline=False,
                   title=dict(font=dict(size=12, color=C.COLORS["slate"]))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=12), title=dict(font=dict(size=12))),
        margin=dict(l=10, r=10, t=56, b=10),
        hoverlabel=dict(bgcolor=C.COLORS["ink"], font=dict(color="white", size=12),
                        bordercolor=C.COLORS["ink"]),
        separators=".,",
    )
    pio.templates["osmi"] = template
    pio.templates.default = "osmi"


def style_fig(fig: go.Figure, height: int = 380, title: str | None = None,
              legend_top: bool = True) -> go.Figure:
    """Apply the shared sizing and title conventions to any figure."""
    fig.update_layout(height=height, template="osmi")
    if title:
        fig.update_layout(title=dict(text=title))
    if not legend_top:
        fig.update_layout(legend=dict(orientation="v", x=1.02, y=1, yanchor="top"))
    return fig


PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

# Streamlit renamed use_container_width to width="stretch" in 1.49 and warns
# loudly on the old name. Resolving it once here keeps every call site clean and
# keeps the app working on older installs.
_VERSION = tuple(int(p) for p in st.__version__.split(".")[:2] if p.isdigit())
FULL = {"width": "stretch"} if _VERSION >= (1, 49) else {"use_container_width": True}


def show(fig: go.Figure, **kwargs) -> None:
    st.plotly_chart(fig, config=PLOTLY_CONFIG, **FULL, **kwargs)


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {{
        --ink: {C.COLORS['ink']};
        --ink-soft: {C.COLORS['ink_soft']};
        --slate: {C.COLORS['slate']};
        --canvas: {C.COLORS['canvas']};
        --surface: {C.COLORS['surface']};
        --line: {C.COLORS['line']};
        --teal: {C.COLORS['teal']};
        --teal-soft: {C.COLORS['teal_soft']};
        --ochre: {C.COLORS['ochre']};
        --plum: {C.COLORS['plum']};
        --rose: {C.COLORS['rose']};
        --sky: {C.COLORS['sky']};
    }}

    html, body, [class*="st-"], .stMarkdown, p, div, span, label {{
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }}
    h1, h2, h3, h4 {{
        font-family: 'Sora', system-ui, sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.02em;
    }}

    .stApp {{ background: var(--canvas); }}
    .stMain .block-container {{ padding: 2.1rem 2.6rem 4rem; max-width: 1480px; }}
    #MainMenu, footer, header [data-testid="stStatusWidget"] {{ visibility: hidden; }}

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {{
        background: var(--ink);
        border-right: 1px solid rgba(255,255,255,0.06);
    }}
    [data-testid="stSidebar"] * {{ color: #E6EBF0; }}
    [data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}

    .brand {{
        display: flex; align-items: center; gap: 0.7rem;
        padding: 0.2rem 0 1.1rem 0; border-bottom: 1px solid rgba(255,255,255,0.09);
        margin-bottom: 1.1rem;
    }}
    .brand-mark {{
        width: 38px; height: 38px; border-radius: 11px; flex: none;
        background: linear-gradient(140deg, var(--teal), #17A398);
        display: grid; place-items: center; color: #fff;
        font-family: 'Sora'; font-weight: 700; font-size: 1.02rem;
    }}
    .brand-name {{
        font-family: 'Sora'; font-weight: 600; font-size: 0.99rem;
        line-height: 1.22; color: #fff;
    }}
    .brand-sub {{ font-size: 0.72rem; color: #8FA3B4; letter-spacing: 0.02em; }}

    .nav-label {{
        font-size: 0.7rem; letter-spacing: 0.09em; color: #7E93A6;
        margin: 1.1rem 0 0.35rem; font-weight: 600;
    }}

    /* Turn the radio group into a navigation list */
    [data-testid="stSidebar"] [role="radiogroup"] {{ gap: 0.1rem; }}
    [data-testid="stSidebar"] [role="radiogroup"] label {{
        padding: 0.5rem 0.7rem; border-radius: 9px; width: 100%;
        transition: background 0.14s ease, color 0.14s ease;
        font-size: 0.9rem; cursor: pointer; margin: 0;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
        background: rgba(255,255,255,0.055);
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
        background: rgba(15,118,110,0.34);
        box-shadow: inset 2px 0 0 0 #2FB3A6;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{
        color: #fff !important; font-weight: 600;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}
    [data-testid="stSidebar"] [role="radiogroup"] p {{
        color: #BFCDD9 !important; font-size: 0.895rem;
    }}

    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div {{
        background: rgba(255,255,255,0.07); border-color: rgba(255,255,255,0.13);
    }}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
        font-size: 0.79rem; color: #9DB0C0 !important; font-weight: 500;
    }}

    /* ---------- Page header ---------- */
    .page-head {{ margin-bottom: 1.5rem; }}
    .page-eyebrow {{
        display: inline-flex; align-items: center; gap: 0.45rem;
        color: var(--teal); font-weight: 600; font-size: 0.78rem;
        background: #E3F0EF; padding: 0.25rem 0.66rem; border-radius: 999px;
        margin-bottom: 0.7rem;
    }}
    .page-title {{
        font-family: 'Sora'; font-size: 1.82rem; font-weight: 600;
        color: var(--ink); line-height: 1.2; margin: 0 0 0.42rem;
    }}
    .page-sub {{
        color: var(--slate); font-size: 0.96rem; max-width: 74ch; line-height: 1.55;
    }}

    /* ---------- Cards ---------- */
    .card {{
        background: var(--surface); border: 1px solid var(--line);
        border-radius: 14px; padding: 1.15rem 1.3rem; height: 100%;
    }}
    .kpi {{
        background: var(--surface); border: 1px solid var(--line);
        border-radius: 14px; padding: 1rem 1.15rem 1.05rem;
        border-left: 3px solid var(--teal); height: 100%;
    }}
    .kpi-label {{
        color: var(--slate); font-size: 0.79rem; font-weight: 500;
        display: block; margin-bottom: 0.42rem;
    }}
    .kpi-value {{
        font-family: 'Sora'; font-size: 1.92rem; font-weight: 600; color: var(--ink);
        line-height: 1; font-variant-numeric: tabular-nums;
    }}
    .kpi-unit {{ font-size: 1.02rem; color: var(--slate); font-weight: 500; margin-left: 2px; }}
    .kpi-caption {{
        color: var(--slate); font-size: 0.775rem; margin-top: 0.5rem; line-height: 1.42;
    }}
    .kpi-chip {{
        display: inline-block; font-size: 0.73rem; font-weight: 600;
        padding: 0.11rem 0.45rem; border-radius: 6px; margin-top: 0.5rem;
    }}

    /* ---------- Callouts ---------- */
    .insight {{
        background: #FFFFFF; border: 1px solid var(--line);
        border-left: 3px solid var(--ochre); border-radius: 12px;
        padding: 0.92rem 1.12rem; margin: 0.25rem 0 1.1rem;
        font-size: 0.9rem; color: var(--ink-soft); line-height: 1.58;
    }}
    .insight b {{ color: var(--ink); }}
    .insight-title {{
        font-family: 'Sora'; font-weight: 600; color: var(--ink);
        font-size: 0.87rem; display: block; margin-bottom: 0.28rem;
    }}
    .caveat {{ border-left-color: var(--rose); }}
    .method {{ border-left-color: var(--slate); }}

    .section-rule {{
        display: flex; align-items: baseline; gap: 0.6rem;
        margin: 1.9rem 0 0.9rem; padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--line);
    }}
    .section-rule h3 {{ font-size: 1.06rem !important; margin: 0 !important; font-weight: 600; }}
    .section-rule span {{ color: var(--slate); font-size: 0.84rem; }}

    /* ---------- Tabs, tables, misc ---------- */
    .stTabs [data-baseweb="tab-list"] {{ gap: 0.3rem; border-bottom: 1px solid var(--line); }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.89rem; font-weight: 500; padding: 0.55rem 0.9rem; color: var(--slate);
    }}
    .stTabs [aria-selected="true"] {{ color: var(--teal) !important; font-weight: 600; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background: var(--teal); }}

    [data-testid="stMetricValue"] {{ font-family: 'Sora'; }}
    .stDataFrame {{ border-radius: 12px; border: 1px solid var(--line); }}
    div[data-testid="stExpander"] details {{
        border: 1px solid var(--line); border-radius: 12px; background: var(--surface);
    }}
    .stButton button, .stDownloadButton button {{
        border-radius: 9px; border: 1px solid var(--line); font-weight: 500;
    }}
    .stDownloadButton button {{ background: var(--teal); color: #fff; border: none; }}

    .footnote {{
        color: var(--slate); font-size: 0.77rem; line-height: 1.5;
        border-top: 1px solid var(--line); padding-top: 0.75rem; margin-top: 2.2rem;
    }}
    </style>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------

def page_header(eyebrow: str, title: str, subtitle: str, icon: str = "◆") -> None:
    st.markdown(f"""
    <div class="page-head">
        <div class="page-eyebrow">{icon} {eyebrow}</div>
        <div class="page-title">{title}</div>
        <div class="page-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def section(title: str, note: str = "") -> None:
    st.markdown(f"""<div class="section-rule"><h3>{title}</h3><span>{note}</span></div>""",
                unsafe_allow_html=True)


def kpi(label: str, value: str, caption: str = "", accent: str = "teal",
        chip: str | None = None, chip_tone: str = "teal", unit: str = "") -> None:
    chip_colors = {"teal": ("#E3F0EF", C.COLORS["teal"]),
                   "rose": ("#FAE8EA", C.COLORS["rose"]),
                   "ochre": ("#FBF1DC", "#8A5E0F"),
                   "slate": ("#EDF1F4", C.COLORS["slate"])}
    bg, fg = chip_colors.get(chip_tone, chip_colors["teal"])
    chip_html = (f'<div class="kpi-chip" style="background:{bg};color:{fg}">{chip}</div>'
                 if chip else "")
    st.markdown(f"""
    <div class="kpi" style="border-left-color:{C.COLORS[accent]}">
        <span class="kpi-label">{label}</span>
        <span class="kpi-value">{value}<span class="kpi-unit">{unit}</span></span>
        {chip_html}
        <div class="kpi-caption">{caption}</div>
    </div>
    """, unsafe_allow_html=True)


def insight(body: str, title: str = "What this shows", tone: str = "") -> None:
    st.markdown(f"""<div class="insight {tone}">
        <span class="insight-title">{title}</span>{body}</div>""", unsafe_allow_html=True)


def footnote(text: str) -> None:
    st.markdown(f'<div class="footnote">{text}</div>', unsafe_allow_html=True)
