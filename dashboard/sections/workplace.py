"""Employer support: what is provided, what is known about, and what it correlates with."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import charts as X
from dashboard import data_loader as DL
from dashboard import theme as T
from src import analysis as A
from src import config as C

SUPPORT_ITEMS = ["benefits", "care_options", "wellness_program", "seek_help", "anonymity"]


def render(df: pd.DataFrame, full: pd.DataFrame) -> None:
    T.page_header(
        "Employer support", "Provision, and whether anyone knows about it",
        "Five questions ask what an employer offers. The scoring here treats \"I don't know\" as a "
        "zero on purpose: a benefit nobody can confirm is not doing any work. The gap between "
        "those two readings is the most actionable finding in the survey.",
        T.ICON["workplace"])

    if not DL.guard(df):
        return

    mean_support = df["support_score"].mean()
    strong = (df["support_score"] >= 3).mean() * 100
    unaware = np.mean([(df[c] == "Don't know").mean() for c in SUPPORT_ITEMS]) * 100
    anonymity_dk = (df["anonymity"] == "Don't know").mean() * 100

    cols = st.columns(4, gap="small")
    with cols[0]:
        T.kpi("Mean support score", f"{mean_support:.2f}", unit="/5", accent="teal",
              caption="Average number of provisions an employee can confirm.")
    with cols[1]:
        T.kpi("Well supported", f"{strong:.0f}", unit="%", accent="teal",
              chip="score 3 or more", chip_tone="teal",
              caption="Can confirm at least three of the five.")
    with cols[2]:
        T.kpi("Average uncertainty", f"{unaware:.0f}", unit="%", accent="ochre",
              chip="answered 'don't know'", chip_tone="ochre",
              caption="Across the five provision questions.")
    with cols[3]:
        T.kpi("Unsure about anonymity", f"{anonymity_dk:.0f}", unit="%", accent="rose",
              chip="the worst-known policy", chip_tone="rose",
              caption="Cannot say whether using support would stay private.")

    tabs = st.tabs(["Provision", "The support score", "Medical leave", "Who gets support"])

    # -- Provision ---------------------------------------------------------
    with tabs[0]:
        T.show(_provision_matrix(df))
        T.insight(
            f"Anonymity is the standout. Only {(df['anonymity'] == 'Yes').mean() * 100:.0f}% can "
            f"confirm it is protected, while {anonymity_dk:.0f}% simply do not know — the largest "
            "uncertainty on any question in the survey. Anonymity is also the cheapest of the five "
            "to fix: it costs a sentence in a policy document and an email, where benefits cost "
            "money.",
            title="The cheapest gap to close")

        st.markdown("")
        picked = st.selectbox("Break a provision down by another variable",
                              SUPPORT_ITEMS, format_func=lambda c: C.QUESTION_TEXT[c])
        by = st.radio("Split by", ["no_employees", "region", "tech_company", "gender",
                                   "remote_work"],
                      format_func=lambda c: {"no_employees": "Company size", "region": "Region",
                                             "tech_company": "Tech employer", "gender": "Gender",
                                             "remote_work": "Remote work"}[c],
                      horizontal=True)
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            T.show(X.stacked_share(df, by, picked, C.QUESTION_TEXT[picked], height=400,
                                   stack_order=["Yes", "Don't know", "Not sure", "No"]))
        with right:
            T.show(X.rate_bar(df, picked, "Treatment rate by this answer", min_n=20, height=400))

    # -- Support score -----------------------------------------------------
    with tabs[1]:
        left, right = st.columns(2, gap="large")
        with left:
            counts = df["support_score"].value_counts().sort_index()
            fig = go.Figure(go.Bar(
                x=counts.index.astype(str), y=counts.values,
                marker_color=[C.SEQ_TEAL[min(5, int(i) + 1)] for i in counts.index],
                text=[f"{v}<br>{v / counts.sum() * 100:.0f}%" for v in counts.values],
                textposition="outside",
                hovertemplate="score %{x}: %{y} respondents<extra></extra>"))
            fig.update_layout(xaxis_title="Provisions the employee can confirm (0–5)",
                              yaxis_title="Respondents")
            T.show(T.style_fig(fig, height=390, title="Support score distribution"))
        with right:
            rates = A.rate_by_group(df, "support_score", min_n=10)
            T.show(X.line_with_band(rates, "level", "rate", "ci_low", "ci_high",
                                    "Treatment rate rises with confirmable support",
                                    "Support score", "% who sought treatment", height=390))

        left, right = st.columns(2, gap="large")
        with left:
            grouped = df.groupby("support_score")[["openness_score", "stigma_score"]].mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=grouped.index, y=grouped["openness_score"],
                                     mode="lines+markers", name="Openness (0–4)",
                                     line=dict(color=C.COLORS["teal"], width=2.6),
                                     marker=dict(size=9),
                                     hovertemplate="score %{x}<br>openness %{y:.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=grouped.index, y=grouped["stigma_score"],
                                     mode="lines+markers", name="Expected penalty (0–5)",
                                     line=dict(color=C.COLORS["rose"], width=2.6),
                                     marker=dict(size=9, symbol="square"),
                                     hovertemplate="score %{x}<br>stigma %{y:.2f}<extra></extra>"))
            fig.update_layout(xaxis_title="Support score", yaxis_title="Mean index value")
            T.show(T.style_fig(fig, height=390,
                               title="More support, more openness, less expected penalty"))
        with right:
            pivot = df.pivot_table(index="support_score", columns="obs_consequence",
                                   values="stigma_score", aggfunc="mean")
            T.show(X.heatmap(pivot, "Expected penalty by support and witnessed consequences",
                             "Mean stigma score", height=390,
                             scale=C.SEQ_WARM))
            st.caption("Columns: has the respondent seen a coworker face consequences for being open?")

        T.insight(
            "The two lines cross in the middle of the range. Below a score of 2, people expect more "
            "penalty than they are willing to disclose; above it, the relationship inverts. "
            "<b>Causality is unresolved and this data cannot settle it</b> — supportive employers "
            "may attract open people rather than create them — but the association is consistent "
            "across every subgroup tested.",
            title="What the crossing lines can and cannot tell you", tone="caveat")

    # -- Leave -------------------------------------------------------------
    with tabs[2]:
        left, right = st.columns(2, gap="large")
        with left:
            T.show(X.count_bar(df["leave"], "How easy is medical leave for mental health?",
                               order=C.CATEGORY_ORDER["leave"], height=390))
        with right:
            T.show(X.stacked_share(df, "leave", "mental_health_consequence",
                                   "Fear of consequences, by how easy leave is",
                                   height=390, stack_order=["No", "Maybe", "Yes"]))

        left, right = st.columns(2, gap="large")
        with left:
            T.show(X.rate_bar(df, "leave", "Treatment rate by leave difficulty", min_n=20,
                              height=380, horizontal=True))
        with right:
            T.show(X.stacked_share(df, "no_employees", "leave", "Leave clarity by company size",
                                   height=380, stack_order=C.CATEGORY_ORDER["leave"]))

        dk = (df["leave"] == "Don't know").mean() * 100
        T.insight(
            f"<b>{dk:.0f}% of respondents cannot say how hard it would be to take leave</b> — more "
            "than any actual answer. The people who say leave is very difficult and the people who "
            "do not know have almost identical fear profiles, which suggests an unknown policy "
            "functions, behaviourally, like a hostile one.",
            title="Not knowing behaves like a bad answer")

    # -- Who gets support --------------------------------------------------
    with tabs[3]:
        left, right = st.columns(2, gap="large")
        with left:
            pivot = df.pivot_table(index="no_employees", columns="region",
                                   values="support_score", aggfunc="mean")
            pivot = pivot.reindex(C.CATEGORY_ORDER["no_employees"])
            pivot = pivot[[c for c in ["United States", "Europe (other)", "Canada"]
                           if c in pivot.columns]]
            T.show(X.heatmap(pivot, "Mean support score by company size and region",
                             "Support score", height=400))
        with right:
            counts = df.groupby(["no_employees", "region"]).size().unstack(fill_value=0)
            counts = counts.reindex(C.CATEGORY_ORDER["no_employees"])
            counts = counts[[c for c in ["United States", "Europe (other)", "Canada"]
                             if c in counts.columns]]
            T.show(X.heatmap(counts, "Sample size behind each cell", "Respondents", height=400,
                             scale=C.SEQ_WARM))

        T.insight(
            "Always read the left panel against the right one. A cell covering eleven people will "
            "happily display a mean of 4.2 and mean nothing at all. This pairing is deliberate: "
            "every heatmap in this dashboard has a sample-size companion or an n in the tooltip.",
            title="Read these two together", tone="method")

        st.markdown("")
        T.show(X.box_split(df, "support_score", "region",
                           "Support score distribution by region", "Support score (0–5)",
                           height=380))


def _provision_matrix(df: pd.DataFrame) -> go.Figure:
    rows = []
    for col in SUPPORT_ITEMS:
        share = df[col].value_counts(normalize=True) * 100
        rows.append({
            "question": C.QUESTION_TEXT[col][:58],
            "Yes": share.get("Yes", 0),
            "Not sure / don't know": share.get("Don't know", 0) + share.get("Not sure", 0),
            "No": share.get("No", 0),
        })
    mat = pd.DataFrame(rows).set_index("question")
    palette = {"Yes": C.COLORS["teal"], "Not sure / don't know": C.COLORS["ochre"],
               "No": C.COLORS["rose"]}
    fig = go.Figure()
    for answer, color in palette.items():
        fig.add_bar(y=mat.index, x=mat[answer], orientation="h", name=answer, marker_color=color,
                    text=[f"{v:.0f}%" if v >= 7 else "" for v in mat[answer]],
                    textposition="inside", insidetextfont=dict(color="white", size=12),
                    hovertemplate="%{y}<br>" + answer + ": %{x:.1f}%<extra></extra>")
    fig.update_layout(barmode="stack", xaxis_title="Share of respondents", xaxis_range=[0, 100],
                      yaxis=dict(autorange="reversed"))
    return T.style_fig(fig, height=400,
                       title="What employers provide, as employees understand it")
