"""Who answered: age, gender, employment setting, and how each relates to treatment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import charts as X
from dashboard import data_loader as DL
from dashboard import theme as T
from src import config as C


def render(df: pd.DataFrame, full: pd.DataFrame) -> None:
    T.page_header(
        "Respondents", "Who is in this sample",
        "Every rate in this dashboard is a rate among these people. The sample is young, male and "
        "American, so who answered shapes what the answers mean — this page is the calibration.",
        T.ICON["people"])

    if not DL.guard(df):
        return

    cols = st.columns(4, gap="small")
    with cols[0]:
        T.kpi("Median age", f"{df['Age'].median():.0f}", accent="teal",
              chip=f"IQR {df['Age'].quantile(.25):.0f}–{df['Age'].quantile(.75):.0f}",
              chip_tone="teal", caption="Half the sample is under 31.")
    with cols[1]:
        T.kpi("Male", f"{(df['gender'] == 'Male').mean() * 100:.0f}", unit="%", accent="sky",
              chip=f"{(df['gender'] == 'Female').mean() * 100:.0f}% female", chip_tone="slate",
              caption="Skew is steeper than the industry it describes.")
    with cols[2]:
        T.kpi("Work in tech", f"{(df['tech_company'] == 'Yes').mean() * 100:.0f}", unit="%",
              accent="plum", caption="The rest work in tech roles at non-tech employers.")
    with cols[3]:
        T.kpi("Remote 50%+", f"{(df['remote_work'] == 'Yes').mean() * 100:.0f}", unit="%",
              accent="ochre", caption="In 2014, well before remote work was normalised.")

    tabs = st.tabs(["Age and gender", "Employment setting", "Response timing"])

    # -- Age and gender ----------------------------------------------------
    with tabs[0]:
        left, right = st.columns([1.25, 1], gap="large")
        with left:
            T.show(_pyramid(df))
        with right:
            T.show(X.donut(df["gender"], "Gender", height=400))

        left, right = st.columns(2, gap="large")
        with left:
            T.show(X.histogram_split(df, "Age", "treatment", "Age distribution by treatment status",
                                     "Age", height=380))
        with right:
            T.show(X.box_split(df, "Age", "gender", "Age spread within each gender group",
                               "Age", height=380))

        left, right = st.columns(2, gap="large")
        with left:
            T.show(X.rate_bar(df, "age_group", "Treatment rate by age band", min_n=20, height=380))
        with right:
            T.show(X.rate_bar(df, "gender", "Treatment rate by gender", min_n=20, height=380))

        T.insight(
            "Treatment rate climbs steadily with age — from about 45% under 25 to nearly 60% at "
            "40–49. That is what you would expect from a cumulative lifetime measure: the question "
            "asks whether someone has <b>ever</b> sought treatment, so older respondents have had "
            "more years in which to answer yes. Read it as exposure time, not as older workers "
            "struggling more.",
            title="Why the age gradient is not what it looks like")

        T.insight(
            "The gender gap is the second-largest association in the whole survey and it runs in "
            "the direction the clinical literature would predict: women in this sample seek "
            "treatment far more often than men. Nothing here distinguishes a difference in "
            "underlying prevalence from a difference in willingness to seek help — the survey "
            "measures only the second.",
            title="On the gender gap", tone="caveat")

    # -- Employment --------------------------------------------------------
    with tabs[1]:
        left, right = st.columns(2, gap="large")
        with left:
            T.show(X.count_bar(df["no_employees"], "Company size",
                               order=C.CATEGORY_ORDER["no_employees"], height=380))
        with right:
            T.show(X.rate_bar(df, "no_employees", "Treatment rate by company size", min_n=20,
                              height=380))

        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            T.show(X.donut(df["tech_company"], "Employer is primarily tech", height=320))
        with c2:
            T.show(X.donut(df["remote_work"], "Works remotely 50%+ of the time", height=320))
        with c3:
            T.show(X.donut(df["self_employed"], "Self-employed", height=320))

        left, right = st.columns(2, gap="large")
        with left:
            pivot = df.pivot_table(index="no_employees", columns="tech_company",
                                   values="support_score", aggfunc="mean")
            pivot = pivot.reindex(C.CATEGORY_ORDER["no_employees"])
            pivot.columns = ["Non-tech employer" if c == "No" else "Tech employer"
                             for c in pivot.columns]
            T.show(X.grouped_bar(pivot, "Mean support score by size and sector",
                                 "Support score (0–5)", height=400,
                                 colors=[C.COLORS["plum"], C.COLORS["teal"]],
                                 text_fmt="{:.1f}"))
        with right:
            sub = df[df["gender"].isin(["Male", "Female"])]
            pivot = sub.pivot_table(index="support_band", columns="gender",
                                    values="treatment_flag", aggfunc="mean", observed=True) * 100
            T.show(X.grouped_bar(pivot, "Treatment rate by support level and gender",
                                 "% who sought treatment", height=400,
                                 colors=[C.COLORS["plum"], C.COLORS["sky"]]))

        T.insight(
            "Support rises with headcount, which is unsurprising — a 5-person company rarely has an "
            "employee assistance programme. The interesting line is the comparison by sector: "
            "<b>tech employers score at or below non-tech employers at most sizes</b>, in a survey "
            "whose respondents are overwhelmingly tech workers. The industry's reputation for "
            "generous perks does not show up in mental health provision.",
            title="Size explains more than sector")

    # -- Timing ------------------------------------------------------------
    with tabs[2]:
        daily = df.groupby("survey_date").size().reset_index(name="responses")
        daily["survey_date"] = pd.to_datetime(daily["survey_date"])
        daily = daily.sort_values("survey_date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["survey_date"], y=daily["responses"], mode="lines",
                                 fill="tozeroy", line=dict(color=C.COLORS["teal"], width=2.2),
                                 fillcolor="rgba(15,118,110,0.16)",
                                 hovertemplate="%{x|%d %b %Y}<br>%{y} responses<extra></extra>"))
        fig.update_layout(xaxis_title="Date", yaxis_title="Responses")
        T.show(T.style_fig(fig, height=360, title="Responses over time"))

        left, right = st.columns(2, gap="large")
        with left:
            early = df.sort_values("Timestamp").head(len(df) // 2)
            late = df.sort_values("Timestamp").tail(len(df) // 2)
            compare = pd.DataFrame({
                "Measure": ["Sought treatment", "Reports a condition", "Support score ≥3",
                            "Expects consequences"],
                "First half of responses": [
                    early["treatment_flag"].mean() * 100, early["has_condition"].mean() * 100,
                    (early["support_score"] >= 3).mean() * 100,
                    (early["mental_health_consequence"] == "Yes").mean() * 100],
                "Second half": [
                    late["treatment_flag"].mean() * 100, late["has_condition"].mean() * 100,
                    (late["support_score"] >= 3).mean() * 100,
                    (late["mental_health_consequence"] == "Yes").mean() * 100],
            })
            T.show(X.dumbbell(compare, "Measure", "First half of responses", "Second half",
                              "Early versus late respondents", "% of group", height=360,
                              left_name="First half", right_name="Second half"))
        with right:
            T.show(X.rate_bar(df, "has_comment", "Treatment rate by whether they left a comment",
                              min_n=20, height=360))

        T.insight(
            "Almost all responses arrived within days of launch, so this is a snapshot rather than "
            "a time series — no trend can be read from the date axis. The early-versus-late "
            "comparison is a non-response check: if the people who answered first differed sharply "
            "from those who answered later, that would hint at selection bias building over the "
            "collection window. Here they track closely.",
            title="Why timing still gets a page", tone="method")


def _pyramid(df: pd.DataFrame) -> go.Figure:
    sub = df[df["gender"].isin(["Male", "Female"])].dropna(subset=["age_group"])
    pivot = pd.crosstab(sub["age_group"], sub["gender"])
    pivot = pivot.reindex(C.CATEGORY_ORDER["age_group"]).fillna(0)
    for col in ("Male", "Female"):
        if col not in pivot:
            pivot[col] = 0
    fig = go.Figure()
    fig.add_bar(y=pivot.index.astype(str), x=-pivot["Male"], orientation="h", name="Male",
                marker_color=C.COLORS["sky"], customdata=pivot["Male"],
                hovertemplate="%{y} · Male<br>%{customdata} respondents<extra></extra>",
                text=pivot["Male"].astype(int), textposition="inside",
                insidetextfont=dict(color="white"))
    fig.add_bar(y=pivot.index.astype(str), x=pivot["Female"], orientation="h", name="Female",
                marker_color=C.COLORS["plum"],
                hovertemplate="%{y} · Female<br>%{x} respondents<extra></extra>",
                text=pivot["Female"].astype(int), textposition="inside",
                insidetextfont=dict(color="white"))
    fig.update_layout(barmode="relative", xaxis=dict(showticklabels=False, title=""),
                      yaxis=dict(autorange="reversed", title=""))
    return T.style_fig(fig, height=400, title="Age and gender structure of the sample")
