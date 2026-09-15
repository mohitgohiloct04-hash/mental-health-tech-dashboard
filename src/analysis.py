"""
Statistical analysis for the Mental Health in Tech survey.

Three questions drive this module:

1. Which survey answers are genuinely associated with having sought treatment,
   and how strong is each association? (chi-square + Cramer's V)
2. How much does each factor move the odds, holding the others constant?
   (logistic regression with odds ratios)
3. How much of the outcome is predictable at all, and from what?
   (random forest with permutation importance and an honest test score)

Run:  python -m src.analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as C

# Predictors offered to the models. work_interfere is deliberately excluded
# from the "policy" model further down: it is a symptom of having a condition,
# so including it leaks the outcome and drowns out everything actionable.
MODEL_FEATURES_CAT = [
    "gender", "age_group", "region", "no_employees", "self_employed", "remote_work",
    "tech_company", "family_history", "benefits", "care_options", "wellness_program",
    "seek_help", "anonymity", "leave", "mental_health_consequence", "coworkers",
    "supervisor", "mental_health_interview", "mental_vs_physical", "obs_consequence",
]
MODEL_FEATURES_NUM = ["Age", "support_score", "openness_score", "stigma_score"]

LEAKY_FEATURES = ["work_interfere", "work_interfere_rank", "has_condition"]


# --------------------------------------------------------------------------
# 1. Association testing
# --------------------------------------------------------------------------

def cramers_v(confusion: np.ndarray) -> float:
    """Bias-corrected Cramer's V for a contingency table."""
    chi2 = stats.chi2_contingency(confusion, correction=False)[0]
    n = confusion.sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else np.nan


def effect_label(v: float) -> str:
    if pd.isna(v):
        return "undefined"
    if v < 0.10:
        return "negligible"
    if v < 0.20:
        return "small"
    if v < 0.35:
        return "moderate"
    return "strong"


def association_table(df: pd.DataFrame, target: str = "treatment") -> pd.DataFrame:
    """Chi-square test of every categorical column against the target."""
    candidates = [
        "gender", "age_group", "region", "self_employed", "family_history", "work_interfere",
        "no_employees", "remote_work", "tech_company", "benefits", "care_options",
        "wellness_program", "seek_help", "anonymity", "leave", "mental_health_consequence",
        "phys_health_consequence", "coworkers", "supervisor", "mental_health_interview",
        "phys_health_interview", "mental_vs_physical", "obs_consequence", "support_band",
    ]
    rows = []
    for col in candidates:
        if col not in df.columns:
            continue
        table = pd.crosstab(df[col], df[target])
        if table.shape[0] < 2 or table.values.sum() == 0:
            continue
        chi2, p, dof, expected = stats.chi2_contingency(table)
        small_expected = float((expected < 5).mean())
        rows.append({
            "feature": col,
            "question": C.QUESTION_TEXT.get(col, col.replace("_", " ").capitalize()),
            "chi2": round(float(chi2), 2),
            "dof": int(dof),
            "p_value": float(p),
            "cramers_v": round(cramers_v(table.values), 3),
            "effect": effect_label(cramers_v(table.values)),
            "n": int(table.values.sum()),
            "low_expected_cells": round(small_expected, 3),
            "leaky": col in LEAKY_FEATURES,
        })
    out = pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)

    # Benjamini-Hochberg correction: 24 simultaneous tests would otherwise
    # produce roughly one spurious "significant" result by chance alone.
    m = len(out)
    order = out["p_value"].rank(method="first").astype(int)
    out["p_adjusted"] = (out["p_value"] * m / order).clip(upper=1.0)
    out["significant"] = out["p_adjusted"] < 0.05
    return out


def rate_by_group(df: pd.DataFrame, col: str, target_flag: str = "treatment_flag",
                  min_n: int = 1) -> pd.DataFrame:
    """Treatment rate per level of `col`, with a Wilson confidence interval."""
    grouped = df.groupby(col, observed=True)[target_flag].agg(["sum", "count"])
    grouped = grouped[grouped["count"] >= min_n]
    rate = grouped["sum"] / grouped["count"]
    z = 1.96
    n = grouped["count"]
    denom = 1 + z**2 / n
    centre = (rate + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denom
    return pd.DataFrame({
        "level": grouped.index.astype(str),
        "n": n.values,
        "rate": (rate * 100).round(1).values,
        "ci_low": ((centre - margin) * 100).clip(0, 100).round(1).values,
        "ci_high": ((centre + margin) * 100).clip(0, 100).round(1).values,
    })


def parity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compare the mental and physical versions of the paired questions."""
    rows = []
    for mental, physical, label in C.PARITY_PAIRS:
        for answer in ["Yes", "Maybe", "No"]:
            rows.append({
                "question": label,
                "answer": answer,
                "Mental health": round((df[mental] == answer).mean() * 100, 1),
                "Physical health": round((df[physical] == answer).mean() * 100, 1),
            })
    out = pd.DataFrame(rows)
    out["gap"] = (out["Mental health"] - out["Physical health"]).round(1)
    return out


# --------------------------------------------------------------------------
# 2 & 3. Models
# --------------------------------------------------------------------------

def _build_matrix(df: pd.DataFrame):
    features = MODEL_FEATURES_CAT + MODEL_FEATURES_NUM
    X = df[features].copy()
    for col in MODEL_FEATURES_CAT:
        X[col] = X[col].astype(str)
    X["Age"] = X["Age"].fillna(X["Age"].median())
    y = df["treatment_flag"]
    return X, y


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), MODEL_FEATURES_CAT),
        ("num", StandardScaler(), MODEL_FEATURES_NUM),
    ])


def fit_models(df: pd.DataFrame) -> dict:
    """Fit both models and return everything the dashboard needs to show."""
    X, y = _build_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=C.RANDOM_STATE, stratify=y)

    logit = Pipeline([("prep", _preprocessor()),
                      ("clf", LogisticRegression(max_iter=2000, C=0.5,
                                                 random_state=C.RANDOM_STATE))])
    forest = Pipeline([("prep", _preprocessor()),
                       ("clf", RandomForestClassifier(n_estimators=500, max_depth=9,
                                                      min_samples_leaf=6,
                                                      class_weight="balanced",
                                                      random_state=C.RANDOM_STATE, n_jobs=-1))])
    results = {}
    for name, model in (("Logistic regression", logit), ("Random forest", forest)):
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_test, proba)
        cv = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 3),
            "f1": round(float(f1_score(y_test, pred)), 3),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
            "cv_auc_mean": round(float(cv.mean()), 3),
            "cv_auc_std": round(float(cv.std()), 3),
            "confusion": confusion_matrix(y_test, pred).tolist(),
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        }

    # Odds ratios from the logistic model: the interpretable half of the story.
    feature_names = logit.named_steps["prep"].get_feature_names_out()
    coefs = logit.named_steps["clf"].coef_[0]
    odds = pd.DataFrame({
        "term": [n.split("__", 1)[1] for n in feature_names],
        "coefficient": coefs,
        "odds_ratio": np.exp(coefs),
    })
    odds["direction"] = np.where(odds["odds_ratio"] >= 1, "Raises odds", "Lowers odds")
    odds["abs_effect"] = (odds["odds_ratio"] - 1).abs()
    results["odds_ratios"] = odds.sort_values("abs_effect", ascending=False).reset_index(drop=True)

    # Permutation importance on held-out data: what the forest actually relies
    # on, measured by how much performance drops when a column is shuffled.
    perm = permutation_importance(forest, X_test, y_test, n_repeats=15,
                                  random_state=C.RANDOM_STATE, scoring="roc_auc", n_jobs=-1)
    results["importance"] = pd.DataFrame({
        "feature": X.columns,
        "importance": perm.importances_mean,
        "std": perm.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    results["test_size"] = int(len(y_test))
    results["train_size"] = int(len(y_train))
    return results


# --------------------------------------------------------------------------
# Correlation between the engineered indices
# --------------------------------------------------------------------------

def index_correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Age", "company_size_rank", "support_score", "openness_score", "stigma_score",
            "leave_ease", "parity_gap", "work_interfere_rank", "treatment_flag"]
    labels = {"Age": "Age", "company_size_rank": "Company size", "support_score": "Support score",
              "openness_score": "Openness score", "stigma_score": "Stigma score",
              "leave_ease": "Leave is easy", "parity_gap": "Parity gap",
              "work_interfere_rank": "Work interference", "treatment_flag": "Sought treatment"}
    corr = df[cols].corr(method="spearman")
    corr.index = [labels[c] for c in corr.index]
    corr.columns = [labels[c] for c in corr.columns]
    return corr.round(3)


def main() -> None:
    df = pd.read_csv(C.CLEAN_DATA)

    assoc = association_table(df)
    assoc.to_csv(C.TABLES / "association_tests.csv", index=False)
    print("Top associations with having sought treatment")
    print(assoc[["feature", "cramers_v", "effect", "p_adjusted", "significant"]].head(12)
          .to_string(index=False))

    parity_summary(df).to_csv(C.TABLES / "parity_gap.csv", index=False)
    index_correlations(df).to_csv(C.TABLES / "index_correlations.csv")

    res = fit_models(df)
    res["importance"].to_csv(C.TABLES / "feature_importance.csv", index=False)
    res["odds_ratios"].to_csv(C.TABLES / "odds_ratios.csv", index=False)
    print("\nModel performance")
    for name in ("Logistic regression", "Random forest"):
        m = res[name]
        print(f"  {name:22s} AUC {m['roc_auc']:.3f} | CV AUC {m['cv_auc_mean']:.3f}"
              f" +/- {m['cv_auc_std']:.3f} | accuracy {m['accuracy']:.3f}")
    print("\nTop predictors (permutation importance)")
    print(res["importance"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
