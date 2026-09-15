"""Treatment drivers: every association, tested, ranked and explorable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from dashboard import charts as X
from dashboard import data_loader as DL
from dashboard import theme as T
from src import analysis as A
from src import config as C


def render(df: pd.DataFrame, full: pd.DataFrame) -> None:
    T.page_header(
        "Treatment drivers", "What separates the half who sought help",
        "Every categorical question in the survey is tested against the outcome with a chi-square "
        "test, ranked by effect size rather than p-value, and corrected for the fact that running "
        "24 tests at once will produce a false positive by chance.",
        T.ICON["drivers"])

    if not DL.guard(df, 60):
        return

    assoc = DL.associations(df)
    actionable = assoc[~assoc["leaky"]]

    cols = st.columns(4, gap="small")
    with cols[0]:
        T.kpi("Tests run", f"{len(assoc)}", accent="slate",
              caption="One chi-square per categorical question.")
    with cols[1]:
        T.kpi("Significant after correction", f"{int(assoc['significant'].sum())}", accent="teal",
              chip="Benjamini-Hochberg", chip_tone="teal",
              caption="False discovery rate held at 5%.")
    with cols[2]:
        top = actionable.iloc[0]
        T.kpi("Strongest driver", f"{top['cramers_v']:.2f}", accent="plum",
              chip=top["feature"], chip_tone="slate",
              caption="Cramer's V, excluding symptom measures.")
    with cols[3]:
        T.kpi("Negligible effects", f"{int((assoc['cramers_v'] < 0.1).sum())}", accent="ochre",
              caption="Statistically detectable, practically irrelevant.")

    tabs = st.tabs(["Ranked associations", "Explore one variable", "Segments", "The leakage trap"])

    # -- Ranked ------------------------------------------------------------
    with tabs[0]:
        show_leaky = st.toggle("Include work interference (a symptom of the outcome)", value=False)
        frame = assoc if show_leaky else actionable
        top = frame.head(14).copy()
        top["label"] = top["question"].str.slice(0, 56)
        top["color"] = np.where(top["leaky"], C.COLORS["slate"],
                                np.where(top["cramers_v"] >= 0.2, C.COLORS["teal"],
                                         C.COLORS["sky"]))
        T.show(X.lollipop(top, "label", "cramers_v",
                          "Effect size of every question against having sought treatment",
                          "Cramer's V", height=520, color_col="color"))

        T.insight(
            "Cramer's V is used here instead of p-values because with 1,259 responses almost "
            "everything is 'significant'. <b>Significance says an effect is probably real; effect "
            "size says whether it is worth anything.</b> Conventional reading: below 0.10 "
            "negligible, 0.10–0.20 small, 0.20–0.35 moderate, above that strong.",
            title="Why the ranking uses effect size", tone="method")

        display = frame[["question", "cramers_v", "effect", "chi2", "dof", "p_value",
                         "p_adjusted", "n", "significant"]].copy()
        display.columns = ["Question", "Cramer's V", "Effect", "Chi-square", "df", "p",
                           "p (corrected)", "n", "Significant"]
        display["p"] = display["p"].apply(_fmt_p)
        display["p (corrected)"] = display["p (corrected)"].apply(_fmt_p)
        st.dataframe(display, hide_index=True, **T.FULL, height=440)
        st.download_button("Download the full test table",
                           frame.to_csv(index=False).encode(),
                           "association_tests.csv", "text/csv")

    # -- Explore one -------------------------------------------------------
    with tabs[1]:
        options = [c for c in assoc["feature"] if c in df.columns]
        picked = st.selectbox("Variable", options,
                              format_func=lambda c: C.QUESTION_TEXT.get(
                                  c, c.replace("_", " ").capitalize()))
        row = assoc[assoc["feature"] == picked].iloc[0]

        cols = st.columns(4, gap="small")
        with cols[0]:
            T.kpi("Cramer's V", f"{row['cramers_v']:.3f}", accent="teal",
                  chip=row["effect"], chip_tone="teal", caption="Effect size.")
        with cols[1]:
            T.kpi("Chi-square", f"{row['chi2']:.1f}", accent="slate",
                  chip=f"df = {row['dof']}", chip_tone="slate", caption="Test statistic.")
        with cols[2]:
            T.kpi("Corrected p", _fmt_p(row["p_adjusted"]), accent="plum",
                  chip="significant" if row["significant"] else "not significant",
                  chip_tone="teal" if row["significant"] else "rose",
                  caption="After Benjamini-Hochberg correction.")
        with cols[3]:
            T.kpi("Low-count cells", f"{row['low_expected_cells'] * 100:.0f}", unit="%",
                  accent="ochre" if row["low_expected_cells"] > 0 else "slate",
                  caption="Cells with expected count under 5 — chi-square is unreliable above 20%.")

        left, right = st.columns(2, gap="large")
        with left:
            T.show(X.rate_bar(df, picked, "Treatment rate by answer", min_n=15, height=400))
        with right:
            T.show(X.stacked_share(df, picked, "treatment", "Composition within each answer",
                                   height=400, stack_order=["Yes", "No"]))

        left, right = st.columns(2, gap="large")
        with left:
            observed = pd.crosstab(df[picked], df["treatment"])
            T.show(X.heatmap(observed, "Observed counts", "Respondents", height=380,
                             text=observed.astype(str)))
        with right:
            chi2, p, dof, expected = stats.chi2_contingency(observed)
            residuals = pd.DataFrame((observed.values - expected) / np.sqrt(expected),
                                     index=observed.index, columns=observed.columns)
            T.show(X.heatmap(residuals.round(2), "Standardised residuals",
                             "Deviation from expected", height=380,
                             scale=C.DIVERGING[::-1], zmid=0,
                             text=residuals.round(1).astype(str)))

        T.insight(
            "The residual panel is where the test result becomes a sentence. A cell above +2 means "
            "that combination appears markedly more often than independence would predict; below "
            "−2, markedly less. The chi-square statistic tells you the table as a whole is "
            "unusual — the residuals tell you which corner of it is doing the work.",
            title="Reading the residuals", tone="method")

    # -- Segments ----------------------------------------------------------
    with tabs[2]:
        c1, c2 = st.columns(2)
        row_var = c1.selectbox("Rows", ["no_employees", "region", "support_band", "leave",
                                        "gender", "age_group"], index=0,
                               format_func=_label)
        col_var = c2.selectbox("Columns", ["age_group", "gender", "support_band", "region",
                                           "family_history", "remote_work"], index=0,
                               format_func=_label)

        _segments(df, row_var, col_var)

    # -- Leakage -----------------------------------------------------------
    with tabs[3]:
        st.markdown("")
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            sub = df[df["work_interfere"] != "No condition reported"]
            T.show(X.stacked_share(sub, "work_interfere", "treatment",
                                   "Treatment status by reported work interference",
                                   height=400, stack_order=["Yes", "No"]))
        with right:
            T.show(X.sunburst(df, ["family_history", "treatment", "work_interfere"],
                              "Family history → treatment → interference", height=400))

        T.insight(
            "Work interference has a Cramer's V of 0.69 against treatment — more than double the "
            "next strongest factor. That looks like the headline finding until you notice the "
            "survey only asked the question of people who reported a condition. <b>It is a "
            "restatement of the outcome, not an explanation of it</b>, and any model that includes "
            "it will score beautifully while telling you nothing you can act on.",
            title="The trap in this dataset", tone="caveat")

        st.markdown("")
        comparison = pd.DataFrame({
            "Model": ["With work interference", "Without (used in this dashboard)"],
            "What it learns": [
                "That people with a disruptive condition seek treatment — true, circular, useless.",
                "That family history, care-option awareness, gender and stigma move the odds.",
            ],
            "Actionable?": ["No", "Yes"],
        })
        st.dataframe(comparison, hide_index=True, **T.FULL)



def _segments(df: pd.DataFrame, row_var: str, col_var: str) -> None:
    if row_var == col_var:
        st.info("Rows and columns are the same variable — pick two different ones to cross.")
        return

    rate = df.pivot_table(index=row_var, columns=col_var, values="treatment_flag",
                          aggfunc="mean", observed=True) * 100
    counts = df.pivot_table(index=row_var, columns=col_var, values="treatment_flag",
                            aggfunc="count", observed=True)
    if row_var in C.CATEGORY_ORDER:
        rate = rate.reindex([i for i in C.CATEGORY_ORDER[row_var] if i in rate.index])
        counts = counts.reindex([i for i in C.CATEGORY_ORDER[row_var] if i in counts.index])
    if col_var in C.CATEGORY_ORDER:
        keep = [c for c in C.CATEGORY_ORDER[col_var] if c in rate.columns]
        rate, counts = rate[keep], counts[keep]

    masked = rate.where(counts >= 10)
    text = masked.round(0).astype("Float64").astype(str).replace("<NA>", "–")
    text = text.replace(r"\.0$", "%", regex=True) + "\nn=" + counts.astype("Int64").astype(str)
    T.show(X.heatmap(masked, f"Treatment rate: {_label(row_var)} by {_label(col_var)}",
                     "% who sought treatment", height=460, text=text))
    st.caption("Cells with fewer than 10 respondents are left blank rather than shown as a "
               "percentage that would move by 10 points per person.")

    left, right = st.columns(2, gap="large")
    with left:
        T.show(X.rate_bar(df, row_var, f"Treatment rate by {_label(row_var).lower()}",
                          min_n=15, height=380))
    with right:
        T.show(X.rate_bar(df, col_var, f"Treatment rate by {_label(col_var).lower()}",
                          min_n=15, height=380))


def _label(col: str) -> str:
    return {"no_employees": "Company size", "region": "Region", "support_band": "Support level",
            "leave": "Leave difficulty", "gender": "Gender", "age_group": "Age band",
            "family_history": "Family history", "remote_work": "Remote work"}.get(
        col, col.replace("_", " ").capitalize())


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"
