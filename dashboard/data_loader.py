"""
Data access layer for the dashboard.

Everything expensive — reading the CSV, running 24 chi-square tests, fitting
two models — happens once and is cached. Sections then work with plain frames.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src import analysis as A
from src import config as C
from src import data_cleaning as DC


@st.cache_data(show_spinner=False)
def load_clean() -> pd.DataFrame:
    """Load the processed survey, rebuilding it from raw if it is missing."""
    if not C.CLEAN_DATA.exists():
        raw = pd.read_csv(C.RAW_DATA)
        df, log = DC.clean(raw)
        df.to_csv(C.CLEAN_DATA, index=False)
        C.CLEANING_LOG.write_text(json.dumps(log, indent=2))
    else:
        df = pd.read_csv(C.CLEAN_DATA)
    df["age_group"] = pd.Categorical(df["age_group"], C.CATEGORY_ORDER["age_group"], ordered=True)
    df["support_band"] = pd.Categorical(df["support_band"],
                                        ["None (0)", "Partial (1-2)", "Strong (3-5)"], ordered=True)
    return df


@st.cache_data(show_spinner=False)
def load_raw() -> pd.DataFrame:
    return pd.read_csv(C.RAW_DATA)


@st.cache_data(show_spinner=False)
def load_cleaning_log() -> dict:
    if C.CLEANING_LOG.exists():
        return json.loads(C.CLEANING_LOG.read_text())
    return {"steps": []}


@st.cache_data(show_spinner=False)
def associations(df: pd.DataFrame) -> pd.DataFrame:
    return A.association_table(df)


@st.cache_resource(show_spinner=False)
def models(_df: pd.DataFrame, cache_key: int):
    """Fit both models. cache_key is the row count so filtered views refit."""
    return A.fit_models(_df)


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------

FILTER_DEFAULTS = {
    "regions": "All regions",
    "genders": "All",
    "sizes": "All sizes",
    "age": (18, 72),
    "tech_only": False,
    "condition_only": False,
}


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    out = df.copy()
    if f.get("regions"):
        out = out[out["region"].isin(f["regions"])]
    if f.get("genders"):
        out = out[out["gender"].isin(f["genders"])]
    if f.get("sizes"):
        out = out[out["no_employees"].isin(f["sizes"])]
    lo, hi = f.get("age", (0, 100))
    out = out[out["Age"].between(lo, hi) | out["Age"].isna()]
    if f.get("tech_only"):
        out = out[out["tech_company"] == "Yes"]
    if f.get("condition_only"):
        out = out[out["has_condition"]]
    return out


def filter_summary(df: pd.DataFrame, filtered: pd.DataFrame) -> str:
    share = len(filtered) / len(df) * 100 if len(df) else 0
    return f"{len(filtered):,} of {len(df):,} responses in view ({share:.0f}%)"


def guard(filtered: pd.DataFrame, minimum: int = 30) -> bool:
    """Stop a section from drawing rates off a handful of rows."""
    if len(filtered) < minimum:
        st.warning(
            f"Only {len(filtered)} responses match the current filters. "
            f"Rates below {minimum} responses are too noisy to read, so this view is paused — "
            "widen the filters in the sidebar.", icon="⚠")
        return False
    return True
