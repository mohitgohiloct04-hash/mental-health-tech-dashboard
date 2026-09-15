"""Geography: where respondents are, and how support and treatment vary by place."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import charts as X
from dashboard import data_loader as DL
from dashboard import theme as T
from src import analysis as A
from src import config as C
from src.data_cleaning import MIN_COUNTRY_N


def render(df: pd.DataFrame, full: pd.DataFrame) -> None:
    T.page_header(
        "Geography", "Where the responses came from",
        "Sixty per cent of the sample is American and the median country contributed two "
        "responses. Every map here suppresses places below a reportable sample size rather than "
        "colouring them in, because a country with three respondents will produce a 100% rate.",
        T.ICON["map"])

    if not DL.guard(df):
        return

    cols = st.columns(4, gap="small")
    with cols[0]:
        T.kpi("Countries", f"{df['Country'].nunique()}", accent="teal",
              caption="Represented in the sample.")
    with cols[1]:
        T.kpi("United States", f"{(df['Country'] == 'United States').mean() * 100:.0f}", unit="%",
              accent="plum", chip=f"{int((df['Country'] == 'United States').sum())} responses",
              chip_tone="slate", caption="The single dominant country.")
    with cols[2]:
        reportable = df["Country"].value_counts()
        T.kpi("Reportable countries", f"{int((reportable >= MIN_COUNTRY_N).sum())}", accent="ochre",
              chip=f"n ≥ {MIN_COUNTRY_N}", chip_tone="ochre",
              caption="The rest are shown in counts only, never as rates.")
    with cols[3]:
        us = df[df["Country"] == "United States"]
        T.kpi("US states", f"{us['state'].nunique()}", accent="sky",
              caption="With at least one respondent.")

    tabs = st.tabs(["World", "United States", "Region comparison"])

    with tabs[0]:
        left, right = st.columns([1.35, 1], gap="large")
        with left:
            T.show(X.choropleth(df, f"Treatment rate by country (n ≥ {MIN_COUNTRY_N})",
                                height=470, min_n=MIN_COUNTRY_N))
        with right:
            T.show(X.count_bar(df["Country"], "Responses by country (top 12)",
                               order=df["Country"].value_counts().head(12).index.tolist(),
                               horizontal=True, height=470, color=C.COLORS["teal"]))

        T.insight(
            "Grey countries are not zeros — they are places with too few respondents to report a "
            "rate. This is the difference between a map that informs and a map that invents: "
            f"below {MIN_COUNTRY_N} responses, a single person moves the percentage by more than "
            "six points.",
            title="Why most of the map is empty", tone="method")

        rates = A.rate_by_group(df, "Country", min_n=MIN_COUNTRY_N)
        rates = rates.sort_values("rate", ascending=False)
        display = rates.copy()
        display["95% CI"] = display.apply(lambda r: f"{r['ci_low']:.0f}–{r['ci_high']:.0f}%", axis=1)
        display = display[["level", "n", "rate", "95% CI"]]
        display.columns = ["Country", "Responses", "% sought treatment", "95% CI"]
        st.dataframe(display, hide_index=True, **T.FULL)
        st.caption("The confidence intervals overlap heavily. Differences between these countries "
                   "are not distinguishable from sampling noise at this sample size.")

    with tabs[1]:
        us = df[df["Country"] == "United States"]
        if len(us) < 40:
            st.info("Not enough US respondents in the current filter to draw the state view.")
        else:
            left, right = st.columns([1.35, 1], gap="large")
            with left:
                T.show(X.us_state_map(df, "Treatment rate by US state (n ≥ 8)", height=440))
            with right:
                T.show(X.count_bar(us["state"], "Responses by state (top 12)",
                                   order=us["state"].value_counts().head(12).index.tolist(),
                                   horizontal=True, height=440, color=C.COLORS["plum"]))

            left, right = st.columns(2, gap="large")
            with left:
                T.show(X.rate_bar(us, "state", "Treatment rate in the best-represented states",
                                  min_n=25, height=380))
            with right:
                pivot = us.groupby("state")["support_score"].agg(["mean", "count"])
                pivot = pivot[pivot["count"] >= 15].sort_values("mean", ascending=False)
                fig = go.Figure(go.Bar(
                    y=pivot.index[::-1], x=pivot["mean"][::-1], orientation="h",
                    marker_color=C.COLORS["teal"], customdata=pivot["count"][::-1],
                    text=[f"{v:.2f}" for v in pivot["mean"][::-1]], textposition="auto",
                    hovertemplate="%{y}<br>mean support %{x:.2f}<br>n=%{customdata}<extra></extra>"))
                fig.update_layout(xaxis_title="Mean support score (0–5)")
                T.show(T.style_fig(fig, height=380, title="Employer support by state (n ≥ 15)"))

    with tabs[2]:
        left, right = st.columns(2, gap="large")
        with left:
            T.show(X.rate_bar(df, "region", "Treatment rate by region", min_n=25, height=400))
        with right:
            T.show(X.box_split(df, "support_score", "region",
                               "Employer support by region", "Support score (0–5)", height=400))

        measures = pd.DataFrame({
            "region": sorted(df["region"].unique()),
        })
        measures["Sought treatment"] = [
            df[df["region"] == r]["treatment_flag"].mean() * 100 for r in measures["region"]]
        measures["Expects consequences"] = [
            (df[df["region"] == r]["mental_health_consequence"] == "Yes").mean() * 100
            for r in measures["region"]]
        measures["Support score ≥3"] = [
            (df[df["region"] == r]["support_score"] >= 3).mean() * 100 for r in measures["region"]]
        measures["Unclear leave policy"] = [
            (df[df["region"] == r]["leave"] == "Don't know").mean() * 100
            for r in measures["region"]]
        counts = df["region"].value_counts()
        measures = measures[measures["region"].map(counts) >= 25]

        pivot = measures.set_index("region").round(1)
        T.show(X.heatmap(pivot, "Regional comparison across four measures", "% of respondents",
                         height=380, text=pivot.round(0).astype(int).astype(str) + "%"))

        T.insight(
            "American respondents report both the highest treatment rate and the most confirmable "
            "employer support — consistent with a system where employer-provided health cover is "
            "the main route to care. European respondents, most of whom have public healthcare, "
            "report less employer provision and a lower treatment rate here. <b>The survey cannot "
            "separate an access effect from a reporting effect</b>, and this comparison should be "
            "treated as a question rather than a finding.",
            title="A tempting comparison to over-read", tone="caveat")
