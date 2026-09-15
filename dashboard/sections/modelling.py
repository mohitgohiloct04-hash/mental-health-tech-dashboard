"""Predictive modelling: how much of the outcome is predictable, and from what."""

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
        "Modelling", "How predictable is any of this?",
        "Two models are fitted on the full survey: logistic regression for interpretable odds, and "
        "a random forest for whatever non-linear structure the linear model misses. Both exclude "
        "work interference, which would leak the answer.",
        T.ICON["model"])

    with st.spinner("Fitting models…"):
        res = DL.models(full, len(full))

    forest, logit = res["Random forest"], res["Logistic regression"]

    cols = st.columns(4, gap="small")
    with cols[0]:
        T.kpi("Random forest AUC", f"{forest['roc_auc']:.3f}", accent="teal",
              chip=f"CV {forest['cv_auc_mean']:.3f} ± {forest['cv_auc_std']:.3f}", chip_tone="teal",
              caption="On a held-out quarter of the data.")
    with cols[1]:
        T.kpi("Logistic AUC", f"{logit['roc_auc']:.3f}", accent="plum",
              chip=f"CV {logit['cv_auc_mean']:.3f} ± {logit['cv_auc_std']:.3f}", chip_tone="slate",
              caption="Nearly identical — the structure here is mostly linear.")
    with cols[2]:
        T.kpi("Accuracy", f"{forest['accuracy'] * 100:.1f}", unit="%", accent="sky",
              chip="baseline 50.6%", chip_tone="slate",
              caption="Against an almost perfectly balanced outcome.")
    with cols[3]:
        T.kpi("Held-out responses", f"{res['test_size']}", accent="slate",
              chip=f"{res['train_size']} used to train", chip_tone="slate",
              caption="Stratified split, fixed random seed.")

    tabs = st.tabs(["Performance", "What moves the odds", "What the forest relies on",
                    "Try a profile"])

    # -- Performance -------------------------------------------------------
    with tabs[0]:
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            fig = go.Figure()
            for name, color in (("Random forest", C.COLORS["teal"]),
                                ("Logistic regression", C.COLORS["plum"])):
                fig.add_trace(go.Scatter(x=res[name]["roc"]["fpr"], y=res[name]["roc"]["tpr"],
                                         mode="lines", name=f"{name} ({res[name]['roc_auc']:.3f})",
                                         line=dict(color=color, width=2.6),
                                         hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Coin flip",
                                     line=dict(color=C.COLORS["slate"], dash="dash", width=1.4),
                                     hoverinfo="skip"))
            fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
            T.show(T.style_fig(fig, height=420, title="ROC curves on held-out data"))
        with right:
            cm = np.array(forest["confusion"])
            frame = pd.DataFrame(cm, index=["Actually no", "Actually yes"],
                                 columns=["Predicted no", "Predicted yes"])
            labels = frame.astype(str) + "\n" + (frame / cm.sum() * 100).round(0).astype(int).astype(str) + "%"
            T.show(X.heatmap(frame, "Random forest confusion matrix", "Responses", height=420,
                             text=labels))

        T.insight(
            f"An AUC near {forest['roc_auc']:.2f} means that given one person who sought treatment "
            f"and one who did not, the model ranks them correctly about {forest['roc_auc'] * 100:.0f}% "
            "of the time. That is a genuine signal and a modest one. <b>Roughly three-quarters of "
            "the variation is not explained by anything the survey asked</b> — personal history, "
            "severity, cost, access and a hundred other things the form never touched.",
            title="What an AUC of this size actually means")

        st.markdown("")
        perf = pd.DataFrame([
            {"Model": name, "AUC (held-out)": res[name]["roc_auc"],
             "AUC (5-fold CV)": f"{res[name]['cv_auc_mean']:.3f} ± {res[name]['cv_auc_std']:.3f}",
             "Accuracy": res[name]["accuracy"], "F1": res[name]["f1"]}
            for name in ("Logistic regression", "Random forest")])
        st.dataframe(perf, hide_index=True, **T.FULL)
        st.caption("Cross-validated AUC sits close to the held-out figure for both models, which "
                   "is the check that the single split was not lucky.")

    # -- Odds --------------------------------------------------------------
    with tabs[1]:
        odds = res["odds_ratios"].head(18).copy()
        odds["color"] = np.where(odds["odds_ratio"] >= 1, C.COLORS["teal"], C.COLORS["rose"])
        odds["term"] = odds["term"].str.replace("_", " ").str.slice(0, 46)
        T.show(X.lollipop(odds, "term", "odds_ratio",
                          "Odds ratios from the logistic model",
                          "Odds of having sought treatment", height=560, color_col="color",
                          reference=1.0, value_fmt="{:.2f}×"))

        T.insight(
            "Each bar is the effect of that answer <b>holding every other variable constant</b>, "
            "which is what separates this from the single-variable charts elsewhere. A value of "
            "2.0 means the odds double; 0.5 means they halve. Teal raises the odds of having "
            "sought treatment, rose lowers them.",
            title="Reading an odds ratio", tone="method")

        display = res["odds_ratios"].head(25)[["term", "odds_ratio", "coefficient", "direction"]]
        display.columns = ["Term", "Odds ratio", "Coefficient (log-odds)", "Direction"]
        st.dataframe(display.round(3), hide_index=True, **T.FULL, height=380)
        st.download_button("Download all odds ratios",
                           res["odds_ratios"].to_csv(index=False).encode(),
                           "odds_ratios.csv", "text/csv")

    # -- Importance --------------------------------------------------------
    with tabs[2]:
        imp = res["importance"].head(14).copy()
        imp["feature"] = imp["feature"].str.replace("_", " ")
        imp["color"] = np.where(imp["importance"] > 0.005, C.COLORS["teal"], C.COLORS["slate"])
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            T.show(X.lollipop(imp, "feature", "importance",
                              "Permutation importance on held-out data",
                              "Drop in AUC when the column is shuffled", height=480,
                              color_col="color", value_fmt="{:.3f}"))
        with right:
            st.markdown("<div style='height:56px'></div>", unsafe_allow_html=True)
            T.insight(
                "Permutation importance is measured by shuffling one column in the held-out set and "
                "watching how far performance falls. Unlike the impurity-based importance that "
                "comes free with a forest, it cannot be inflated by a column simply having many "
                "distinct values, and it is measured on data the model never saw.",
                title="Why this and not feature_importances_", tone="method")
            st.dataframe(res["importance"].head(12).round(4), hide_index=True,
                         **T.FULL, height=330)

        T.insight(
            "Family history dominates and it is not something an employer can change. Below it sit "
            "<b>awareness of care options, gender, and the stigma index</b> — and the first and "
            "third of those are policy levers. The ranking is a reasonable order of operations for "
            "anyone deciding where to spend a mental health budget.",
            title="The one ranking worth acting on")

    # -- Predictor ---------------------------------------------------------
    with tabs[3]:
        st.markdown("Set a profile and the fitted model returns its estimated probability. "
                    "This is a way to interrogate the model, not a diagnostic tool.")
        _predictor(full, res)


def _predictor(full: pd.DataFrame, res: dict) -> None:
    from src.analysis import MODEL_FEATURES_CAT, MODEL_FEATURES_NUM, _build_matrix
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    model = _fit_for_prediction(full, len(full))

    c1, c2, c3, c4 = st.columns(4)
    profile = {}
    profile["Age"] = c1.slider("Age", 18, 70, 31)
    profile["gender"] = c2.selectbox("Gender", ["Male", "Female", "Non-binary / other"])
    profile["family_history"] = c3.selectbox("Family history of mental illness", ["No", "Yes"])
    profile["no_employees"] = c4.selectbox("Company size", C.CATEGORY_ORDER["no_employees"],
                                           index=2)

    c1, c2, c3, c4 = st.columns(4)
    profile["benefits"] = c1.selectbox("Employer provides benefits", ["Yes", "No", "Don't know"])
    profile["care_options"] = c2.selectbox("Knows the care options", ["Yes", "No", "Not sure"])
    profile["anonymity"] = c3.selectbox("Anonymity protected", ["Yes", "No", "Don't know"])
    profile["leave"] = c4.selectbox("Medical leave is", C.CATEGORY_ORDER["leave"], index=3)

    c1, c2, c3, c4 = st.columns(4)
    profile["mental_health_consequence"] = c1.selectbox("Expects negative consequences",
                                                        ["No", "Maybe", "Yes"])
    profile["supervisor"] = c2.selectbox("Would tell a supervisor",
                                         C.CATEGORY_ORDER["supervisor"], index=2)
    profile["obs_consequence"] = c3.selectbox("Has seen a coworker penalised", ["No", "Yes"])
    profile["remote_work"] = c4.selectbox("Works remotely 50%+", ["No", "Yes"])

    row = _profile_row(full, profile)
    proba = float(model.predict_proba(row)[0, 1])
    baseline = full["treatment_flag"].mean()

    st.markdown("")
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        tone = "teal" if proba >= baseline else "slate"
        T.kpi("Estimated probability of having sought treatment", f"{proba * 100:.1f}", unit="%",
              accent=tone, chip=f"{(proba - baseline) * 100:+.1f} pts vs the 50.6% baseline",
              chip_tone="teal" if proba >= baseline else "rose",
              caption="Random forest, fitted on all 1,259 responses.")
    with right:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=proba * 100,
            number=dict(suffix="%", font=dict(size=34, family="Sora")),
            gauge=dict(axis=dict(range=[0, 100], tickcolor=C.COLORS["slate"]),
                       bar=dict(color=C.COLORS["teal"], thickness=0.72),
                       bgcolor=C.COLORS["canvas"], borderwidth=0,
                       threshold=dict(line=dict(color=C.COLORS["rose"], width=3),
                                      value=baseline * 100)),
            domain=dict(x=[0, 1], y=[0, 1])))
        T.show(T.style_fig(fig, height=230, title="Model estimate (red mark = sample baseline)"))

    T.insight(
        "Change one field at a time to see what the model has learned. Family history and care-"
        "option awareness move the number most; company size and remote work barely move it at "
        "all. <b>This estimates how likely someone with this profile was to answer yes in a 2014 "
        "survey.</b> It says nothing about any individual's health and should never be used as "
        "though it does.",
        title="How to use this, and how not to", tone="caveat")


@st.cache_resource(show_spinner=False)
def _fit_for_prediction(_df: pd.DataFrame, cache_key: int):
    from src.analysis import _build_matrix, _preprocessor, MODEL_FEATURES_CAT
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    X, y = _build_matrix(_df)
    model = Pipeline([("prep", _preprocessor()),
                      ("clf", RandomForestClassifier(n_estimators=500, max_depth=9,
                                                     min_samples_leaf=6, class_weight="balanced",
                                                     random_state=C.RANDOM_STATE, n_jobs=-1))])
    model.fit(X, y)
    return model


def _profile_row(full: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """Build one model-ready row: user choices where given, sample mode elsewhere."""
    from src.analysis import MODEL_FEATURES_CAT, MODEL_FEATURES_NUM

    row = {}
    for col in MODEL_FEATURES_CAT:
        row[col] = str(profile.get(col, full[col].astype(str).mode().iloc[0]))
    row["Age"] = profile["Age"]

    support_items = {"benefits": "Yes", "care_options": "Yes", "wellness_program": "Yes",
                     "seek_help": "Yes", "anonymity": "Yes"}
    row["support_score"] = sum(1 for c, v in support_items.items() if row.get(c) == v)
    row["openness_score"] = (C.ORDINAL_MAPS["coworkers"].get(row["coworkers"], 1)
                             + C.ORDINAL_MAPS["supervisor"].get(row["supervisor"], 1))
    row["stigma_score"] = (
        C.ORDINAL_MAPS["mental_health_consequence"].get(row["mental_health_consequence"], 1)
        + (2 - C.ORDINAL_MAPS["mental_health_interview"].get(row["mental_health_interview"], 1))
        + (1 if row["obs_consequence"] == "Yes" else 0))
    return pd.DataFrame([row])[MODEL_FEATURES_CAT + MODEL_FEATURES_NUM]
