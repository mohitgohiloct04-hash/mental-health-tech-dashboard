"""Data quality: what the raw export looked like and every decision made on it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import charts as X
from dashboard import data_loader as DL
from dashboard import theme as T
from src import config as C
from src import data_cleaning as DC


def render(df: pd.DataFrame, full: pd.DataFrame) -> None:
    raw = DL.load_raw()
    log = DL.load_cleaning_log()

    T.page_header(
        "Data quality", "What the raw export looked like",
        "The source file is a Google Form export: free-text where the analysis wants categories, "
        "joke answers in the age box, and blanks that mean three different things. Every "
        "transformation is listed here so any number elsewhere in the dashboard can be traced back.",
        T.ICON["quality"])

    cols = st.columns(4, gap="small")
    with cols[0]:
        T.kpi("Raw responses", f"{len(raw):,}", accent="slate",
              caption="No rows were dropped during cleaning.")
    with cols[1]:
        T.kpi("Columns", f"{raw.shape[1]} → {full.shape[1]}", accent="teal",
              chip=f"+{full.shape[1] - raw.shape[1]} engineered", chip_tone="teal",
              caption="Original questions plus derived indices and flags.")
    with cols[2]:
        T.kpi("Cells missing", f"{raw.isna().mean().mean() * 100:.1f}", unit="%", accent="ochre",
              caption="Concentrated in four columns, not spread thin.")
    with cols[3]:
        T.kpi("Impossible ages", f"{len(log.get('invalid_age_values', []))}", accent="rose",
              chip="set to missing", chip_tone="rose",
              caption="Including -1726 and 99,999,999,999.")

    T.section("Missingness", "four columns, three different meanings")

    miss = (raw.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    meaning = {
        "comments": "Optional free-text box. Blank = chose not to elaborate.",
        "state": "Only asked of US respondents. Blank = outside the US.",
        "work_interfere": "Only shown to people reporting a condition. Blank = no condition.",
        "self_employed": "Question added after launch. Blank = answered early.",
    }
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        fig = go.Figure(go.Bar(
            y=miss.index[::-1], x=miss.values[::-1], orientation="h",
            marker_color=[C.COLORS["rose"] if v > 60 else C.COLORS["ochre"]
                          for v in miss.values[::-1]],
            text=[f"{v:.0f}%" for v in miss.values[::-1]], textposition="outside",
            hovertemplate="%{y}: %{x:.1f}% missing<extra></extra>"))
        fig.update_layout(xaxis_title="Share of responses missing", xaxis_range=[0, 100])
        T.show(T.style_fig(fig, height=330, title="Missing values by column"))
    with right:
        st.markdown("<div style='height:52px'></div>", unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame({"Column": miss.index,
                          "Missing": [f"{v:.0f}%" for v in miss.values],
                          "What a blank means": [meaning.get(c, "") for c in miss.index]}),
            hide_index=True, **T.FULL)

    T.insight(
        "Only one of these four is ordinary missing data. <b>Imputing all four with a mode would "
        "have invented 264 people without conditions into the 'Never interferes' bucket</b> and "
        "manufactured US states for European respondents. Knowing the questionnaire's skip logic "
        "matters more than any imputation technique here.",
        title="Why a blanket fillna would break this dataset", tone="caveat")

    T.section("The age field", "eight entries are not outliers, they are nonsense")

    left, right = st.columns(2, gap="large")
    with left:
        clipped = raw["Age"].clip(-50, 120)
        fig = go.Figure(go.Histogram(x=clipped, nbinsx=60, marker_color=C.COLORS["rose"],
                                     hovertemplate="age %{x}: %{y}<extra></extra>"))
        fig.update_layout(xaxis_title="Reported age (clipped to −50…120 to be plottable)",
                          yaxis_title="Respondents")
        T.show(T.style_fig(fig, height=340, title="Before: raw age values"))
        st.caption("Values found in the raw column: "
                   + ", ".join(f"{v:,}" for v in log.get("invalid_age_values", [])))
    with right:
        fig = go.Figure(go.Histogram(x=full["Age"], nbinsx=32, marker_color=C.COLORS["teal"],
                                     hovertemplate="age %{x}: %{y}<extra></extra>"))
        fig.add_vline(x=full["Age"].median(), line_dash="dash", line_color=C.COLORS["ink"],
                      annotation_text=f"median {full['Age'].median():.0f}")
        fig.update_layout(xaxis_title="Validated age", yaxis_title="Respondents")
        T.show(T.style_fig(fig, height=340, title="After: 18 to 72, median 31"))
        st.caption("Ages outside 15–80 were set to missing rather than winsorised — a typo of "
                   "329 carries no information about the true value, so clipping it to 80 would "
                   "fabricate one.")

    T.section("The gender field", "49 spellings, three analytic groups")

    left, right = st.columns([1.4, 1], gap="large")
    with left:
        counts = raw["Gender"].value_counts().head(16)
        fig = go.Figure(go.Bar(y=counts.index[::-1], x=counts.values[::-1], orientation="h",
                               marker_color=C.COLORS["slate"],
                               hovertemplate="'%{y}': %{x} responses<extra></extra>"))
        fig.update_layout(xaxis_title="Respondents", xaxis_type="log")
        T.show(T.style_fig(fig, height=430,
                           title=f"Raw free-text answers (top 16 of {raw['Gender'].nunique()}, log scale)"))
    with right:
        T.show(X.count_bar(full["gender"], "After normalisation",
                           order=C.CATEGORY_ORDER["gender"], height=430))

    with st.expander("See the mapping rules"):
        sample = pd.DataFrame({"Raw answer": raw["Gender"].value_counts().index[:40]})
        sample["Mapped to"] = sample["Raw answer"].apply(DC.normalise_gender)
        sample["Count"] = [raw["Gender"].value_counts()[v] for v in sample["Raw answer"]]
        st.dataframe(sample, hide_index=True, **T.FULL, height=320)
        st.markdown(
            "The non-binary group ends at 13 people. It is reported everywhere it appears but "
            "**no rate is calculated from it** — a percentage over 13 people moves 7.7 points "
            "per person.")

    T.section("Full cleaning log", "every transformation, in order")

    steps = pd.DataFrame(log.get("steps", []))
    if not steps.empty:
        steps.columns = ["Step", "What was done and why", "Rows affected"]
        st.dataframe(steps, hide_index=True, **T.FULL)

    T.section("Engineered features", "the composite indices the analysis runs on")

    derived = pd.DataFrame([
        {"Feature": k, "Definition": v} for k, v in C.DERIVED_TEXT.items()
    ] + [
        {"Feature": "has_condition",
         "Definition": "True if the respondent answered the work-interference question, which the "
                       "form only showed to people reporting a condition."},
        {"Feature": "leave_ease / company_size_rank / work_interfere_rank",
         "Definition": "Ordinal ranks for questions with a natural order, so correlations and "
                       "trend lines are meaningful. 'Don't know' is left as missing, not "
                       "placed in the middle."},
        {"Feature": "age_group / region",
         "Definition": "Bands and geographic groupings used wherever a raw level would fall "
                       "below a reportable sample size."},
    ])
    st.dataframe(derived, hide_index=True, **T.FULL)

    T.section("Browse the raw file", "unmodified source")
    st.dataframe(raw.head(200), **T.FULL, height=320)
    st.caption(f"First 200 of {len(raw):,} rows, exactly as exported.")
