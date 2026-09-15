"""Executive overview: the five numbers that frame everything else."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts as X
from dashboard import data_loader as DL
from dashboard import theme as T
from src import config as C


def render(df: pd.DataFrame, full: pd.DataFrame) -> None:
    T.page_header(
        "Overview", "Mental health in the tech workplace",
        "1,259 people in the software industry answered 26 questions about their own mental "
        "health and how their employer handles it. Half had sought treatment. This dashboard "
        "works out what separates them from the half who had not.",
        T.ICON["overview"])

    if not DL.guard(df):
        return

    rate = df["treatment_flag"].mean() * 100
    condition = df["has_condition"].mean() * 100
    no_support = (df["support_score"] == 0).mean() * 100
    dk_leave = (df["leave"] == "Don't know").mean() * 100
    fear = (df["mental_health_consequence"] == "Yes").mean() * 100

    cols = st.columns(5, gap="small")
    with cols[0]:
        T.kpi("Sought treatment", f"{rate:.1f}", unit="%", accent="teal",
              chip=f"{int(df['treatment_flag'].sum()):,} people", chip_tone="teal",
              caption="Have ever sought help for a mental health condition.")
    with cols[1]:
        T.kpi("Report a condition", f"{condition:.0f}", unit="%", accent="plum",
              chip=f"{int(df['has_condition'].sum()):,} people", chip_tone="slate",
              caption="Answered the follow-up question about work interference.")
    with cols[2]:
        T.kpi("No confirmable support", f"{no_support:.0f}", unit="%", accent="rose",
              chip="support score 0", chip_tone="rose",
              caption="Cannot confirm a single one of five employer provisions.")
    with cols[3]:
        T.kpi("Don't know leave policy", f"{dk_leave:.0f}", unit="%", accent="ochre",
              chip="largest single answer", chip_tone="ochre",
              caption="Cannot say how hard it is to take mental health leave.")
    with cols[4]:
        phys = (df["phys_health_consequence"] == "Yes").mean() * 100
        T.kpi("Expect consequences", f"{fear:.0f}", unit="%", accent="rose",
              chip=f"vs {phys:.0f}% for physical health", chip_tone="rose",
              caption="Believe raising mental health with their employer would backfire.")

    T.section("The shape of the sample", "who answered, and what they reported")

    left, right = st.columns([1, 1.45], gap="large")
    with left:
        T.show(X.donut(df["treatment"], "Have you sought treatment?", height=330))
    with right:
        T.show(X.count_bar(
            df["work_interfere"], "If you have a condition, does it interfere with work?",
            order=C.CATEGORY_ORDER["work_interfere"] + ["No condition reported"], height=330))

    T.insight(
        "The survey only asked about work interference of people who reported a condition, so the "
        f"<b>{int((df['work_interfere'] == 'No condition reported').sum())} blanks are an answer, not "
        "missing data</b>. Treating them as missing — the common shortcut with this dataset — "
        "silently drops a fifth of the sample from every crosstab. They are kept here as an "
        "explicit level.",
        title="A cleaning decision worth knowing about")

    T.section("What actually predicts treatment", "effect size, not just significance")

    assoc = DL.associations(df)
    top = assoc[~assoc["leaky"]].head(10).copy()
    top["label"] = top["question"].str.slice(0, 58)
    top["color"] = top["effect"].map({"strong": C.COLORS["teal"], "moderate": C.COLORS["teal"],
                                      "small": C.COLORS["sky"], "negligible": C.COLORS["slate"]})

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        T.show(X.lollipop(top, "label", "cramers_v",
                          "Association with having sought treatment",
                          "Cramer's V", height=430, color_col="color"))
    with right:
        T.show(X.rate_bar(df, "support_band", "Treatment rate by employer support", min_n=20,
                          height=430))

    T.insight(
        "Cramer's V measures how strongly two categorical variables move together, from 0 to 1. "
        "Family history leads on <b>biology the employer cannot touch</b>. Everything below it — "
        "knowing your care options, having benefits, protected anonymity — is <b>policy the "
        "employer controls</b>. That second group is where the actionable findings live.",
        title="Reading this chart")

    T.section("Two views of the same workplace", "provision versus perception")

    left, right = st.columns(2, gap="large")
    with left:
        support_df = pd.DataFrame({
            "question": [C.QUESTION_TEXT[c][:52] for c in
                         ["benefits", "care_options", "wellness_program", "seek_help", "anonymity"]],
            "Yes": [(df[c] == "Yes").mean() * 100 for c in
                    ["benefits", "care_options", "wellness_program", "seek_help", "anonymity"]],
            "Unclear": [((df[c] == "Don't know") | (df[c] == "Not sure")).mean() * 100 for c in
                        ["benefits", "care_options", "wellness_program", "seek_help", "anonymity"]],
        })
        T.show(X.dumbbell(support_df, "question", "Yes", "Unclear",
                          "Confirmed support versus uncertainty",
                          "% of respondents", height=380,
                          left_name="Can confirm it exists", right_name="Doesn't know"))
    with right:
        parity = pd.DataFrame({
            "measure": ["Expects negative consequences", "Would not raise it in an interview"],
            "Mental health": [(df["mental_health_consequence"] == "Yes").mean() * 100,
                              (df["mental_health_interview"] == "No").mean() * 100],
            "Physical health": [(df["phys_health_consequence"] == "Yes").mean() * 100,
                                (df["phys_health_interview"] == "No").mean() * 100],
        })
        T.show(X.dumbbell(parity, "measure", "Physical health", "Mental health",
                          "The same question, asked about both kinds of health",
                          "% of respondents", height=380,
                          left_name="Physical health", right_name="Mental health"))

    T.insight(
        "On the left, the gap between the two dots is the communication problem: for anonymity, "
        "<b>almost twice as many people are unsure as can confirm it</b>. On the right, the gap is "
        "the stigma problem: the survey asked identical questions about mental and physical health, "
        "and the answers are not close.",
        title="Where the two gaps sit")

    T.footnote(
        "Source: Open Sourcing Mental Illness (OSMI) Mental Health in Tech Survey, 2014. "
        "Self-selected sample; associations are not causal. Percentages respond to the sidebar "
        "filters — the narrative text describes the unfiltered dataset.")
