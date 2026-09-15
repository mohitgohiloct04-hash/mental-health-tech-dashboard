"""
Mental Health in Tech — analytics dashboard.

Run from the project root:

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import data_loader as DL          # noqa: E402
from dashboard import theme as T                 # noqa: E402
from dashboard.sections import (drivers, explorer, geography, insights, modelling,  # noqa: E402
                                overview, people, quality, stigma, workplace)
from src import config as C                      # noqa: E402

st.set_page_config(page_title="Mental Health in Tech — Analytics",
                   page_icon="◆", layout="wide", initial_sidebar_state="expanded")

PAGES = {
    "Overview": (T.ICON["overview"], overview),
    "Respondents": (T.ICON["people"], people),
    "Employer support": (T.ICON["workplace"], workplace),
    "Stigma and openness": (T.ICON["stigma"], stigma),
    "Treatment drivers": (T.ICON["drivers"], drivers),
    "Modelling": (T.ICON["model"], modelling),
    "Geography": (T.ICON["map"], geography),
    "Explorer": (T.ICON["explorer"], explorer),
    "Findings": (T.ICON["insights"], insights),
    "Data quality": (T.ICON["quality"], quality),
}


def sidebar(full):
    with st.sidebar:
        st.markdown("""
        <div class="brand">
          <div class="brand-mark">MH</div>
          <div>
            <div class="brand-name">Mental Health in Tech</div>
            <div class="brand-sub">OSMI survey · 1,259 responses · 2014</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="nav-label">SECTIONS</div>', unsafe_allow_html=True)
        choice = st.radio("Navigation", list(PAGES.keys()),
                          format_func=lambda k: f"{PAGES[k][0]}  {k}",
                          label_visibility="collapsed")

        st.markdown('<div class="nav-label">FILTERS</div>', unsafe_allow_html=True)
        regions = st.multiselect("Region", sorted(full["region"].unique()),
                                 default=sorted(full["region"].unique()))
        genders = st.multiselect("Gender", ["Male", "Female", "Non-binary / other"],
                                 default=["Male", "Female", "Non-binary / other"])
        sizes = st.multiselect("Company size", C.CATEGORY_ORDER["no_employees"],
                               default=C.CATEGORY_ORDER["no_employees"])
        age = st.slider("Age range", 18, 72, (18, 72))
        tech_only = st.checkbox("Tech employers only", value=False)
        condition_only = st.checkbox("Only people reporting a condition", value=False)

        filters = {"regions": regions, "genders": genders, "sizes": sizes, "age": age,
                   "tech_only": tech_only, "condition_only": condition_only}
        filtered = DL.apply_filters(full, filters)

        share = len(filtered) / len(full) * 100
        st.markdown(f"""
        <div style="margin-top:0.9rem;padding:0.7rem 0.85rem;border-radius:10px;
             background:rgba(255,255,255,0.055);font-size:0.79rem;line-height:1.5">
          <div style="font-family:Sora;font-size:1.22rem;color:#fff">{len(filtered):,}</div>
          <div style="color:#93A7B7">of {len(full):,} responses in view ({share:.0f}%)</div>
        </div>
        """, unsafe_allow_html=True)

        if len(filtered) < len(full):
            st.caption("Filters apply to charts on every section. The narrative text describes "
                       "the full dataset.")

        st.markdown("""
        <div style="margin-top:1.4rem;padding-top:0.9rem;
             border-top:1px solid rgba(255,255,255,0.09);font-size:0.72rem;color:#7E93A6;
             line-height:1.55">
          Source: Open Sourcing Mental Illness, Mental Health in Tech Survey 2014.<br>
          Associations shown are not causal.
        </div>
        """, unsafe_allow_html=True)

    return choice, filtered


def main() -> None:
    T.register_template()
    T.inject_css()
    full = DL.load_clean()
    choice, filtered = sidebar(full)
    PAGES[choice][1].render(filtered, full)


if __name__ == "__main__":
    main()
