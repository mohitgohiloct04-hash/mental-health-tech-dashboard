"""Data explorer: build your own crosstab, chart any question, export anything."""

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

CATEGORICAL = ["gender", "age_group", "region", "Country", "no_employees", "self_employed",
               "remote_work", "tech_company", "family_history", "treatment", "work_interfere",
               "benefits", "care_options", "wellness_program", "seek_help", "anonymity", "leave",
               "mental_health_consequence", "phys_health_consequence", "coworkers", "supervisor",
               "mental_health_interview", "phys_health_interview", "mental_vs_physical",
               "obs_consequence", "support_band", "has_condition", "has_comment"]

NUMERIC = ["Age", "support_score", "openness_score", "stigma_score", "parity_gap",
           "leave_ease", "company_size_rank", "comment_length"]


def render(df: pd.DataFrame, full: pd.DataFrame) -> None:
    T.page_header(
        "Explorer", "Ask your own question of the data",
        "Everything in the previous sections is a pre-built view. This one is open: cross any two "
        "questions, chart any single one, or take the cleaned file away and do your own work "
        "with it.",
        T.ICON["explorer"])

    if not DL.guard(df):
        return

    tabs = st.tabs(["Crosstab builder", "Single question", "Numeric relationships", "Table & export"])

    # -- Crosstab ----------------------------------------------------------
    with tabs[0]:
        c1, c2, c3 = st.columns([1, 1, 1])
        row_var = c1.selectbox("Rows", CATEGORICAL, index=CATEGORICAL.index("no_employees"),
                               format_func=_label)
        col_var = c2.selectbox("Columns", CATEGORICAL, index=CATEGORICAL.index("treatment"),
                               format_func=_label)
        mode = c3.selectbox("Show", ["Row percentages", "Counts", "Column percentages",
                                     "Standardised residuals"])

        _crosstab(df, row_var, col_var, mode)

    # -- Single question ---------------------------------------------------
    with tabs[1]:
        question = st.selectbox("Question", CATEGORICAL, format_func=_label, key="single")
        st.caption(C.QUESTION_TEXT.get(question, "Derived field — see the data quality page."))

        left, right = st.columns([1, 1.2], gap="large")
        with left:
            T.show(X.donut(df[question].astype(str), _label(question), height=380))
        with right:
            T.show(X.count_bar(df[question].astype(str), "Responses",
                               order=C.CATEGORY_ORDER.get(question), height=380))

        split = st.radio("Break down by", ["gender", "age_group", "region", "no_employees",
                                           "support_band", "treatment"],
                         format_func=_label, horizontal=True, key="single_split")
        left, right = st.columns(2, gap="large")
        with left:
            T.show(X.stacked_share(df, split, question,
                                   f"{_label(question)} by {_label(split).lower()}", height=420))
        with right:
            observed = pd.crosstab(df[split].astype(str), df[question].astype(str))
            observed = _reorder(observed, split, question)
            T.show(X.heatmap(observed.astype(float), "Counts behind the shares", "Respondents",
                             height=420, text=observed.astype(int).astype(str)))

    # -- Numeric -----------------------------------------------------------
    with tabs[2]:
        c1, c2 = st.columns(2)
        xvar = c1.selectbox("Horizontal", NUMERIC, index=0, format_func=_label)
        yvar = c2.selectbox("Vertical", NUMERIC, index=3, format_func=_label)
        colour = st.radio("Colour by", ["treatment", "gender", "support_band", "region"],
                          format_func=_label, horizontal=True)

        if xvar == yvar:
            st.info("Both axes are the same measure — pick two different ones to see a "
                    "relationship.")
            xvar, yvar = "Age", "stigma_score"

        sub = df[list(dict.fromkeys([xvar, yvar, colour]))].dropna()
        left, right = st.columns([1.3, 1], gap="large")
        with left:
            fig = go.Figure()
            for level in sorted(sub[colour].astype(str).unique()):
                part = sub[sub[colour].astype(str) == level]
                jitter_x = part[xvar] + np.random.uniform(-0.18, 0.18, len(part))
                jitter_y = part[yvar] + np.random.uniform(-0.18, 0.18, len(part))
                fig.add_trace(go.Scatter(
                    x=jitter_x, y=jitter_y, mode="markers", name=str(level),
                    marker=dict(size=7, opacity=0.55,
                                color=C.ANSWER_COLORS.get(level, C.COLORS["teal"]),
                                line=dict(width=0)),
                    hovertemplate=f"{level}<br>{_label(xvar)} %{{x:.0f}}"
                                  f"<br>{_label(yvar)} %{{y:.0f}}<extra></extra>"))
            fig.update_layout(xaxis_title=_label(xvar), yaxis_title=_label(yvar))
            T.show(T.style_fig(fig, height=440,
                               title=f"{_label(yvar)} against {_label(xvar)}"))
            st.caption("Points are jittered so that overlapping whole-number answers stay visible.")
        with right:
            corr = A.index_correlations(df)
            T.show(X.heatmap(corr, "Spearman correlation between indices", "rho", height=440,
                             text=corr.round(2).astype(str), scale=C.DIVERGING[::-1], zmid=0))

        rho, p = stats.spearmanr(sub[xvar], sub[yvar])
        T.insight(
            f"Spearman rho between {_label(xvar).lower()} and {_label(yvar).lower()} is "
            f"<b>{rho:.3f}</b> (p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}). Spearman is used "
            "throughout rather than Pearson because most of these are ordinal scores, where the "
            "distance between 'Rarely' and 'Sometimes' is not guaranteed to equal the distance "
            "between 'Sometimes' and 'Often'.",
            title="The correlation for this pair", tone="method")

    # -- Table -------------------------------------------------------------
    with tabs[3]:
        default_cols = ["Age", "gender", "region", "no_employees", "treatment", "family_history",
                        "support_score", "stigma_score", "openness_score", "leave"]
        chosen = st.multiselect("Columns", df.columns.tolist(), default=default_cols)
        search = st.text_input("Filter rows containing text", "")
        view = df[chosen] if chosen else df
        if search:
            mask = view.astype(str).apply(
                lambda r: r.str.contains(search, case=False, na=False)).any(axis=1)
            view = view[mask]
        st.caption(f"{len(view):,} rows")
        st.dataframe(view, **T.FULL, height=440)

        c1, c2 = st.columns(2)
        c1.download_button("Download the current view", view.to_csv(index=False).encode(),
                           "survey_view.csv", "text/csv", **T.FULL)
        c2.download_button("Download the full cleaned dataset",
                           full.to_csv(index=False).encode(),
                           "survey_clean.csv", "text/csv", **T.FULL)

        with st.expander("Column dictionary"):
            dictionary = pd.DataFrame([
                {"Column": c,
                 "Meaning": C.QUESTION_TEXT.get(c, C.DERIVED_TEXT.get(c, "—")),
                 "Type": str(df[c].dtype),
                 "Distinct values": df[c].nunique()}
                for c in df.columns])
            st.dataframe(dictionary, hide_index=True, **T.FULL, height=420)



def _crosstab(df: pd.DataFrame, row_var: str, col_var: str, mode: str) -> None:
    if row_var == col_var:
        st.info("Pick two different questions.")
        return

    sub = df[[row_var, col_var]].dropna()
    sub[row_var] = sub[row_var].astype(str)
    sub[col_var] = sub[col_var].astype(str)
    observed = pd.crosstab(sub[row_var], sub[col_var])
    observed = _reorder(observed, row_var, col_var)

    if observed.shape[0] < 2 or observed.shape[1] < 2:
        st.info("This pair does not produce a table with at least two rows and two columns.")
        return

    chi2, p, dof, expected = stats.chi2_contingency(observed)
    v = A.cramers_v(observed.values)

    cols = st.columns(4, gap="small")
    with cols[0]:
        T.kpi("Cramer's V", f"{v:.3f}", accent="teal", chip=A.effect_label(v), chip_tone="teal",
              caption="Effect size of the association.")
    with cols[1]:
        T.kpi("Chi-square", f"{chi2:.1f}", accent="slate", chip=f"df = {dof}", chip_tone="slate",
              caption="Test statistic for independence.")
    with cols[2]:
        T.kpi("p-value", "< 0.001" if p < 0.001 else f"{p:.3f}", accent="plum",
              chip="reject independence" if p < 0.05 else "cannot reject",
              chip_tone="teal" if p < 0.05 else "slate",
              caption="Uncorrected — see the drivers page for multiple-test correction.")
    with cols[3]:
        low = (expected < 5).mean() * 100
        T.kpi("Low-count cells", f"{low:.0f}", unit="%", accent="ochre" if low > 20 else "slate",
              chip="test unreliable" if low > 20 else "test valid", chip_tone="rose" if low > 20 else "teal",
              caption="Cells with an expected count below 5.")

    if mode == "Counts":
        matrix, fmt, bar = observed.astype(float), observed.astype(int).astype(str), "Respondents"
        scale, zmid = C.SEQ_TEAL, None
    elif mode == "Row percentages":
        matrix = (observed.div(observed.sum(axis=1), axis=0) * 100).round(1)
        fmt = matrix.round(0).astype(int).astype(str) + "%"
        bar, scale, zmid = "% of row", C.SEQ_TEAL, None
    elif mode == "Column percentages":
        matrix = (observed.div(observed.sum(axis=0), axis=1) * 100).round(1)
        fmt = matrix.round(0).astype(int).astype(str) + "%"
        bar, scale, zmid = "% of column", C.SEQ_WARM, None
    else:
        matrix = pd.DataFrame((observed.values - expected) / np.sqrt(expected),
                              index=observed.index, columns=observed.columns).round(2)
        fmt = matrix.round(1).astype(str)
        bar, scale, zmid = "Deviation from expected", C.DIVERGING[::-1], 0

    T.show(X.heatmap(matrix, f"{_label(row_var)} by {_label(col_var)} — {mode.lower()}",
                     bar, height=max(340, 60 + 46 * len(matrix)), text=fmt,
                     scale=scale, zmid=zmid))

    left, right = st.columns(2, gap="large")
    with left:
        T.show(X.stacked_share(sub, row_var, col_var,
                               f"Composition of {_label(col_var).lower()} within each row",
                               height=400))
    with right:
        if "treatment_flag" in df.columns:
            T.show(X.rate_bar(df, row_var, f"Treatment rate by {_label(row_var).lower()}",
                              min_n=15, height=400))

    st.download_button("Download this crosstab",
                       matrix.to_csv().encode(),
                       f"crosstab_{row_var}_{col_var}.csv", "text/csv")


def _reorder(table: pd.DataFrame, row_var: str, col_var: str) -> pd.DataFrame:
    if row_var in C.CATEGORY_ORDER:
        keep = [i for i in C.CATEGORY_ORDER[row_var] if i in table.index]
        keep += [i for i in table.index if i not in keep]
        table = table.reindex(keep)
    if col_var in C.CATEGORY_ORDER:
        keep = [c for c in C.CATEGORY_ORDER[col_var] if c in table.columns]
        keep += [c for c in table.columns if c not in keep]
        table = table[keep]
    return table


def _label(col: str) -> str:
    pretty = {
        "no_employees": "Company size", "age_group": "Age band", "support_band": "Support level",
        "has_condition": "Reports a condition", "has_comment": "Left a comment",
        "support_score": "Support score", "stigma_score": "Stigma score",
        "openness_score": "Openness score", "parity_gap": "Parity gap",
        "leave_ease": "Leave is easy (rank)", "company_size_rank": "Company size (rank)",
        "comment_length": "Comment length", "Country": "Country", "Age": "Age",
    }
    return pretty.get(col, col.replace("_", " ").capitalize())
