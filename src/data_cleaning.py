"""
Cleaning and feature engineering for the OSMI Mental Health in Tech survey.

The raw file is a Google Form export, which means free-text answers where the
analysis wants categories, joke answers in the age field, and blanks that mean
different things in different columns. Every decision below is recorded in a
cleaning log so the dashboard can show the reader what was changed and why.

Run:  python -m src.data_cleaning
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src import config as C


# --------------------------------------------------------------------------
# Gender normalisation
# --------------------------------------------------------------------------
# 49 distinct spellings for 1,259 responses. The free-text field is collapsed
# to three analytic groups. The non-binary group is small (~18 people), so it
# is reported but never used for a rate comparison without a sample-size note.

_MALE = {
    "male", "m", "maile", "cis male", "mal", "male (cis)", "make", "man", "msle",
    "mail", "malr", "cis man", "male-ish", "something kinda male?", "guy (-ish) ^_^",
    "male leaning androgynous", "ostensibly male, unsure what that really means",
    "male ", "maile ", "cis male ",
}
_FEMALE = {
    "female", "f", "woman", "cis female", "femake", "female (cis)", "cis-female/femme",
    "female (trans)", "trans-female", "trans woman", "femail", "female ",
}
_NONBINARY = {
    "non-binary", "nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender",
    "queer", "queer/she/they", "neuter", "a little about you", "p",
    "male leaning androgynous",
}


def normalise_gender(value: str) -> str:
    """Map a free-text gender answer onto one of three analytic groups."""
    if not isinstance(value, str):
        return "Non-binary / other"
    v = value.strip().lower()
    v = re.sub(r"\s+", " ", v)
    if v in _FEMALE or re.fullmatch(r"(cis[\s-]*)?(fe)?male?\s*\(?trans\)?", v):
        return "Female"
    if v in _MALE:
        return "Male"
    if v in _NONBINARY:
        return "Non-binary / other"
    # Fall back to substring logic for the long tail of one-off spellings.
    if "trans" in v and "female" in v:
        return "Female"
    if v.startswith("f") or "woman" in v or "female" in v:
        return "Female"
    if v.startswith("m") or "man" in v or "male" in v:
        return "Male"
    return "Non-binary / other"


# --------------------------------------------------------------------------
# Region grouping
# --------------------------------------------------------------------------
# 48 countries, but 60% of responses are from the United States. Rates for a
# country with nine respondents are noise, so analysis defaults to regions and
# only shows country-level rates above a minimum sample size.

_EUROPE = {
    "United Kingdom", "Germany", "Netherlands", "Ireland", "France", "Switzerland",
    "Poland", "Italy", "Sweden", "Belgium", "Austria", "Denmark", "Finland", "Norway",
    "Spain", "Portugal", "Russia", "Bulgaria", "Greece", "Croatia", "Czech Republic",
    "Slovenia", "Romania", "Moldova", "Bosnia and Herzegovina", "Latvia", "Hungary",
}
_APAC = {"Australia", "New Zealand", "India", "Singapore", "Japan", "China", "Thailand",
         "Philippines", "Bahamas, The"}

MIN_COUNTRY_N = 15  # below this, a country rate is not reported


def to_region(country: str) -> str:
    if country == "United States":
        return "United States"
    if country == "Canada":
        return "Canada"
    if country in _EUROPE:
        return "Europe (other)"
    if country in _APAC:
        return "Asia-Pacific"
    return "Rest of world"


def age_band(age: float) -> str | float:
    if pd.isna(age):
        return np.nan
    if age < 25:
        return "Under 25"
    if age < 30:
        return "25-29"
    if age < 35:
        return "30-34"
    if age < 40:
        return "35-39"
    if age < 50:
        return "40-49"
    return "50+"


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

AGE_MIN, AGE_MAX = 15, 80


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a cleaned frame plus a log describing every transformation."""
    log: dict = {"input_rows": int(len(df)), "input_columns": int(df.shape[1]), "steps": []}
    out = df.copy()

    def note(step: str, detail: str, affected: int) -> None:
        log["steps"].append({"step": step, "detail": detail, "rows_affected": int(affected)})

    # -- Timestamp ---------------------------------------------------------
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
    out["survey_date"] = out["Timestamp"].dt.date.astype("string")
    out["survey_week"] = out["Timestamp"].dt.to_period("W").astype(str)
    note("Parsed timestamps", "Derived survey_date and survey_week for response-flow analysis.",
         out["Timestamp"].notna().sum())

    # -- Age ---------------------------------------------------------------
    # Contains -1726, 5, 329 and 99,999,999,999. These are nonsense rather than
    # outliers, so they become missing instead of being winsorised.
    bad_age = ~out["Age"].between(AGE_MIN, AGE_MAX)
    log["invalid_age_values"] = sorted(int(v) for v in out.loc[bad_age, "Age"].unique())
    out.loc[bad_age, "Age"] = np.nan
    note("Invalidated impossible ages",
         f"Ages outside {AGE_MIN}-{AGE_MAX} set to missing (joke and typo entries).",
         bad_age.sum())

    out["age_group"] = out["Age"].apply(age_band)
    out["age_group"] = pd.Categorical(out["age_group"],
                                      categories=C.CATEGORY_ORDER["age_group"], ordered=True)

    # -- Gender ------------------------------------------------------------
    raw_gender_levels = out["Gender"].nunique()
    out["gender"] = out["Gender"].apply(normalise_gender)
    note("Collapsed free-text gender",
         f"{raw_gender_levels} raw spellings mapped onto 3 analytic groups.", len(out))

    # -- Geography ---------------------------------------------------------
    out["region"] = out["Country"].apply(to_region)
    # A US state was recorded for non-US respondents in a handful of rows; the
    # field is only meaningful inside the United States.
    stray_state = (out["Country"] != "United States") & out["state"].notna()
    out.loc[stray_state, "state"] = np.nan
    note("Scoped state to US respondents",
         "Cleared US state values recorded against non-US countries.", stray_state.sum())

    # -- self_employed -----------------------------------------------------
    # 18 blanks. The question was added after the survey opened, so blanks are
    # early respondents rather than refusals; filled with the dominant answer
    # and flagged so the imputation stays visible.
    missing_se = out["self_employed"].isna()
    out["self_employed_imputed"] = missing_se
    out["self_employed"] = out["self_employed"].fillna("No")
    note("Imputed self_employed",
         "18 blanks filled with 'No' (modal answer) and flagged in self_employed_imputed.",
         missing_se.sum())

    # -- work_interfere ----------------------------------------------------
    # 264 blanks. This question was only shown to people who reported a
    # condition, so a blank is information, not absence of it. Treating it as
    # missing would silently drop a fifth of the sample from every crosstab.
    missing_wi = out["work_interfere"].isna()
    out["has_condition"] = ~missing_wi
    out["work_interfere"] = out["work_interfere"].fillna("No condition reported")
    note("Reinterpreted blank work_interfere",
         "Blanks mean the respondent reported no condition; kept as an explicit level "
         "and captured in the has_condition flag.", missing_wi.sum())

    # -- Comments ----------------------------------------------------------
    out["has_comment"] = out["comments"].notna()
    out["comment_length"] = out["comments"].fillna("").str.len()
    note("Derived comment features",
         "1,095 of 1,259 left the free-text box empty; kept length and presence as features.",
         out["has_comment"].sum())

    # -- Whitespace hygiene across every remaining object column -----------
    obj_cols = [c for c in out.columns if pd.api.types.is_string_dtype(out[c])]
    for col in obj_cols:
        out[col] = out[col].str.strip()

    # -- Duplicates --------------------------------------------------------
    dupes = out.duplicated(subset=[c for c in out.columns if c != "Timestamp"]).sum()
    note("Checked duplicates", f"{dupes} duplicate response bodies found.", dupes)

    out = engineer_features(out)
    log["output_rows"] = int(len(out))
    log["output_columns"] = int(out.shape[1])
    log["engineered_columns"] = [
        "age_group", "gender", "region", "has_condition", "support_score", "support_band",
        "openness_score", "stigma_score", "parity_gap", "leave_ease", "company_size_rank",
        "work_interfere_rank",
    ]
    return out, log


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the composite indices the analysis is built on."""
    out = df.copy()

    # Support score: how many of five concrete provisions can the respondent
    # actually confirm? "Don't know" scores zero on purpose — a benefit nobody
    # knows about is not a benefit.
    support_items = {"benefits": "Yes", "care_options": "Yes", "wellness_program": "Yes",
                     "seek_help": "Yes", "anonymity": "Yes"}
    out["support_score"] = sum((out[col] == val).astype(int) for col, val in support_items.items())
    out["support_band"] = pd.cut(out["support_score"], bins=[-0.1, 0.5, 2.5, 5.1],
                                 labels=["None (0)", "Partial (1-2)", "Strong (3-5)"])

    # Openness: willingness to talk to the two audiences that matter at work.
    out["openness_score"] = (out["coworkers"].map(C.ORDINAL_MAPS["coworkers"]).fillna(0)
                             + out["supervisor"].map(C.ORDINAL_MAPS["supervisor"]).fillna(0))

    # Stigma: expected cost of being open, from three independent angles.
    out["stigma_score"] = (
        out["mental_health_consequence"].map(C.ORDINAL_MAPS["mental_health_consequence"]).fillna(0)
        + (2 - out["mental_health_interview"].map(C.ORDINAL_MAPS["mental_health_interview"]).fillna(0))
        + (out["obs_consequence"] == "Yes").astype(int)
    )

    # Parity gap: the same question asked about mental and physical health.
    out["parity_gap"] = (
        out["mental_health_consequence"].map(C.ORDINAL_MAPS["mental_health_consequence"])
        - out["phys_health_consequence"].map(C.ORDINAL_MAPS["phys_health_consequence"])
    )
    out["interview_parity_gap"] = (
        out["phys_health_interview"].map(C.ORDINAL_MAPS["phys_health_interview"])
        - out["mental_health_interview"].map(C.ORDINAL_MAPS["mental_health_interview"])
    )

    # Ordinal ranks for correlation work.
    out["leave_ease"] = out["leave"].map(C.ORDINAL_MAPS["leave"])
    out["company_size_rank"] = out["no_employees"].map(C.ORDINAL_MAPS["no_employees"])
    out["work_interfere_rank"] = out["work_interfere"].map(C.ORDINAL_MAPS["work_interfere"])
    out["treatment_flag"] = (out["treatment"] == "Yes").astype(int)

    return out


def main() -> None:
    raw = pd.read_csv(C.RAW_DATA)
    clean_df, log = clean(raw)
    clean_df.to_csv(C.CLEAN_DATA, index=False)
    C.CLEANING_LOG.write_text(json.dumps(log, indent=2))

    print(f"Raw:     {log['input_rows']} rows x {log['input_columns']} cols")
    print(f"Clean:   {log['output_rows']} rows x {log['output_columns']} cols")
    print(f"Saved:   {C.CLEAN_DATA.relative_to(C.ROOT)}")
    print("\nCleaning steps")
    for step in log["steps"]:
        print(f"  - {step['step']}: {step['rows_affected']} rows")


if __name__ == "__main__":
    main()
