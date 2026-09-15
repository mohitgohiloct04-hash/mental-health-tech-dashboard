"""Stigma and openness: what people expect to happen if they speak up."""

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


def render(df: pd.DataFrame, full: pd.DataFrame) -> None:
    T.page_header(
        "Stigma and openness", "The cost people expect for speaking up",
        "The survey asked several questions twice — once about mental health, once about physical "
        "health. The difference between the two answers is the cleanest stigma measurement "
        "available, because the respondent, employer and wording are all held constant.",
        T.ICON["stigma"])

    if not DL.guard(df):
        return

    mh_fear = (df["mental_health_consequence"] == "Yes").mean() * 100
    ph_fear = (df["phys_health_consequence"] == "Yes").mean() * 100
    no_interview = (df["mental_health_interview"] == "No").mean() * 100
    witnessed = (df["obs_consequence"] == "Yes").mean() * 100
    parity_dk = (df["mental_vs_physical"] == "Don't know").mean() * 100

    cols = st.columns(4, gap="small")
    with cols[0]:
        T.kpi("Expect consequences", f"{mh_fear:.0f}", unit="%", accent="rose",
              chip=f"{ph_fear:.0f}% for physical health", chip_tone="rose",
              caption="Believe raising mental health at work would backfire.")
    with cols[1]:
        T.kpi("Would not tell an interviewer", f"{no_interview:.0f}", unit="%", accent="plum",
              chip=f"{(df['phys_health_interview'] == 'No').mean() * 100:.0f}% for physical",
              chip_tone="slate", caption="Would not raise it with a prospective employer.")
    with cols[2]:
        T.kpi("Have seen it happen", f"{witnessed:.0f}", unit="%", accent="ochre",
              caption="Witnessed a coworker face consequences for being open.")
    with cols[3]:
        T.kpi("Can't say if parity exists", f"{parity_dk:.0f}", unit="%", accent="slate",
              caption="Don't know whether their employer treats both equally.")

    T.section("The parity test", "identical questions, two kinds of health")

    left, right = st.columns(2, gap="large")
    with left:
        T.show(_parity_chart(df, "mental_health_consequence", "phys_health_consequence",
                             "Would discussing it with your employer have negative consequences?"))
    with right:
        T.show(_parity_chart(df, "mental_health_interview", "phys_health_interview",
                             "Would you raise it in a job interview?"))

    left, right = st.columns([1, 1.3], gap="large")
    with left:
        gap = df["parity_gap"].dropna()
        counts = gap.value_counts().sort_index()
        labels = {-2: "Fears physical more (−2)", -1: "(−1)", 0: "Treats both the same",
                  1: "(+1)", 2: "Fears mental far more (+2)"}
        colors = {-2: C.COLORS["sky"], -1: "#9FC0D8", 0: C.COLORS["slate"],
                  1: "#D1798A", 2: C.COLORS["rose"]}
        fig = go.Figure(go.Bar(
            x=[labels.get(i, str(i)) for i in counts.index], y=counts.values,
            marker_color=[colors.get(i, C.COLORS["slate"]) for i in counts.index],
            text=[f"{v / counts.sum() * 100:.0f}%" for v in counts.values],
            textposition="outside",
            hovertemplate="%{x}<br>%{y} respondents<extra></extra>"))
        fig.update_layout(yaxis_title="Respondents", xaxis_title="Parity gap")
        T.show(T.style_fig(fig, height=400, title="Distribution of the parity gap"))
    with right:
        T.show(X.stacked_share(df, "region", "mental_vs_physical",
                               "Does your employer take mental health as seriously as physical?",
                               height=400, stack_order=["Yes", "Don't know", "No"]))

    T.insight(
        f"About {(df['parity_gap'] > 0).mean() * 100:.0f}% of respondents fear the mental health "
        f"conversation more than the physical one, against {(df['parity_gap'] < 0).mean() * 100:.0f}% "
        "who fear it less. Because the pair of questions differs only in the word 'mental' or "
        "'physical', this difference cannot be explained by a difficult manager, a bad employer or "
        "a cautious personality — those are held constant inside each respondent.",
        title="Why the paired design matters")

    T.section("Who people will talk to", "coworkers, supervisors, and nobody")

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        T.show(X.count_bar(df["coworkers"], "Would discuss with coworkers",
                           order=C.CATEGORY_ORDER["coworkers"], height=340))
    with c2:
        T.show(X.count_bar(df["supervisor"], "Would discuss with a supervisor",
                           order=C.CATEGORY_ORDER["supervisor"], height=340))
    with c3:
        counts = df["openness_score"].value_counts().sort_index()
        fig = go.Figure(go.Bar(x=counts.index.astype(str), y=counts.values,
                               marker_color=[C.SEQ_TEAL[min(5, int(i) + 1)] for i in counts.index],
                               text=counts.values, textposition="outside",
                               hovertemplate="score %{x}: %{y}<extra></extra>"))
        fig.update_layout(xaxis_title="Openness score (0–4)", yaxis_title="Respondents")
        T.show(T.style_fig(fig, height=340, title="Combined openness"))

    left, right = st.columns(2, gap="large")
    with left:
        pivot = pd.crosstab(df["mental_health_consequence"], df["supervisor"])
        pivot = pivot.reindex(index=["No", "Maybe", "Yes"],
                              columns=C.CATEGORY_ORDER["supervisor"]).fillna(0)
        T.show(X.heatmap(pivot.astype(int), "Fear of consequences versus willingness to talk",
                         "Respondents", height=400,
                         text=pivot.astype(int).astype(str)))
        st.caption("Rows: expects negative consequences. Columns: would discuss with supervisor.")
    with right:
        T.show(X.stacked_share(df, "obs_consequence", "supervisor",
                               "Willingness to talk, by whether they have seen it go badly",
                               height=400, stack_order=C.CATEGORY_ORDER["supervisor"]))
        st.caption("Rows: has seen a coworker face consequences for being open.")

    T.insight(
        "Having personally witnessed a coworker penalised is associated with a sharp drop in "
        "willingness to talk to a supervisor. One visible incident appears to travel further "
        "through an organisation than any number of policy documents — which is an argument for "
        "handling the first disclosure in a team carefully, because everyone else is watching it.",
        title="One visible incident does a lot of work")

    T.section("Openness and treatment", "does silence keep people from help?")

    left, right = st.columns(2, gap="large")
    with left:
        T.show(X.rate_bar(df, "openness_score", "Treatment rate by openness score", min_n=20,
                          height=380))
    with right:
        T.show(X.rate_bar(df, "obs_consequence", "Treatment rate by witnessed consequences",
                          min_n=20, height=380))

    T.insight(
        "Openness at work and seeking treatment are only loosely related, and that is worth "
        "sitting with. People are perfectly capable of getting private treatment while telling "
        "nobody at the office. <b>An employer that sees low disclosure should not conclude its "
        "workforce is well</b> — it may only be concluding that its workforce is discreet.",
        title="Silence is not the same as health", tone="caveat")

    T.section("Stigma index", "three measures combined")

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        T.show(X.box_split(df, "stigma_score", "support_band",
                           "Expected penalty by employer support", "Stigma score (0–5)",
                           height=380))
    with right:
        pivot = df.pivot_table(index="no_employees", columns="support_band",
                               values="stigma_score", aggfunc="mean", observed=True)
        pivot = pivot.reindex(C.CATEGORY_ORDER["no_employees"])
        T.show(X.heatmap(pivot, "Mean stigma score by company size and support level",
                         "Stigma score", height=380, scale=C.SEQ_WARM))

    with st.expander("How the stigma score is built"):
        st.markdown(f"""
        {C.DERIVED_TEXT['stigma_score']}

        - Expects negative consequences at work: **No 0, Maybe 1, Yes 2**
        - Would raise it in an interview: **Yes 0, Maybe 1, No 2** (reversed, so higher is worse)
        - Has witnessed a coworker penalised: **No 0, Yes 1**

        The three components are weighted equally. They correlate with each other at
        moderate strength, which is the intended behaviour for an index: related enough
        to be measuring one underlying thing, distinct enough that each adds information.
        """)


def _parity_chart(df: pd.DataFrame, mental: str, physical: str, title: str) -> go.Figure:
    answers = ["Yes", "Maybe", "No"]
    m = [(df[mental] == a).mean() * 100 for a in answers]
    p = [(df[physical] == a).mean() * 100 for a in answers]
    fig = go.Figure()
    fig.add_bar(x=answers, y=m, name="Mental health", marker_color=C.COLORS["plum"],
                text=[f"{v:.0f}%" for v in m], textposition="outside",
                hovertemplate="Mental health — %{x}: %{y:.1f}%<extra></extra>")
    fig.add_bar(x=answers, y=p, name="Physical health", marker_color=C.COLORS["sky"],
                text=[f"{v:.0f}%" for v in p], textposition="outside",
                hovertemplate="Physical health — %{x}: %{y:.1f}%<extra></extra>")
    fig.update_layout(barmode="group", yaxis_title="% of respondents", yaxis_range=[0, 100])
    return T.style_fig(fig, height=400, title=title)
