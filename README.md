# Mental Health in Tech — end-to-end analysis and dashboard

An exploratory analysis of the 2014 OSMI *Mental Health in Tech* survey (1,259
responses, 48 countries), plus a ten-section Streamlit dashboard built on top of
it.

The guiding question: **half the sample has sought treatment and half has not —
what separates them, and how much of that is something an employer can change?**

---

## Quick start

```bash
pip install -r requirements.txt

python -m src.data_cleaning     # raw → data/processed/survey_clean.csv
python -m src.analysis          # statistical tests + models → reports/tables/
python -m src.eda               # 24 figures + written report → reports/

streamlit run dashboard/app.py
```

The dashboard rebuilds the cleaned file automatically if it is missing, so the
last command works on its own from a fresh clone.

```bash
python tests/test_dashboard.py  # renders all 10 sections headlessly
```

---

## What's in here

```
mental-health-tech/
├── data/
│   ├── raw/survey.csv               # untouched source export
│   └── processed/
│       ├── survey_clean.csv         # 1,259 × 46 — analysis-ready
│       └── cleaning_log.json        # every transformation, with row counts
├── src/
│   ├── config.py                    # paths, palette, question text, ordinal maps
│   ├── data_cleaning.py             # cleaning + feature engineering
│   ├── analysis.py                  # chi-square, Cramér's V, logistic, random forest
│   └── eda.py                       # static figure set + findings report
├── dashboard/
│   ├── app.py                       # shell: navigation and global filters
│   ├── theme.py                     # CSS design system, Plotly template, components
│   ├── charts.py                    # 14 reusable chart builders
│   ├── data_loader.py               # cached loading, filtering, small-sample guards
│   └── sections/                    # ten section modules
├── reports/
│   ├── eda_findings.md              # written findings
│   ├── figures/                     # 24 PNGs
│   └── tables/                      # test results, odds ratios, importances
└── tests/test_dashboard.py          # headless render test for every section
```

---

## The cleaning problems, and how each was handled

This file is a Google Form export, so the interesting work is upfront.

| Problem | Decision |
| --- | --- |
| Ages of −1726, 5, 329, 99,999,999,999 | Set to missing, **not** winsorised. A typo of 329 carries no information about the true value, so clipping it to 80 would fabricate one. 1,251 usable ages remain. |
| 49 free-text gender spellings | Collapsed to Male / Female / Non-binary or other. The third group is 13 people — reported everywhere, but **no rate is ever calculated from it**. |
| 264 blank `work_interfere` | **Not missing data.** The form only showed this question to people reporting a condition, so a blank is an answer. Kept as an explicit level plus a `has_condition` flag. Imputing it would have invented 264 symptom-free people into the "Never" bucket. |
| 515 blank `state` | Only asked of US respondents. Four non-US rows had a state anyway; those were cleared. |
| 18 blank `self_employed` | Question added after launch. Filled with the modal answer and flagged in `self_employed_imputed`. |

## Engineered features

- **`support_score` (0–5)** — how many of five concrete provisions the employee
  can actually confirm. "Don't know" scores zero on purpose: a benefit nobody
  knows about is not doing any work.
- **`openness_score` (0–4)** — willingness to discuss an issue with coworkers
  and with a supervisor.
- **`stigma_score` (0–5)** — expected penalty for openness, from three angles:
  fear of consequences, unwillingness to raise it in an interview, and having
  witnessed a coworker penalised.
- **`parity_gap`** — the survey asked several questions twice, once about mental
  health and once about physical health. The difference is the cleanest stigma
  measure available, because respondent, employer and wording are all held
  constant inside each person.

---

## Method notes

**Effect size over p-values.** With 1,259 responses almost everything clears
p < 0.05. Questions are ranked by bias-corrected Cramér's V, with
Benjamini-Hochberg correction across the 24 simultaneous tests.

**Leakage was the main modelling trap.** `work_interfere` has V = 0.69 against
treatment — double the next strongest factor — because the survey only asked it
of people with a condition. It is a restatement of the outcome, so both models
exclude it. Including it produces a beautiful score and nothing actionable.

**Every rate carries its uncertainty.** Wilson intervals on bar charts, sample
sizes in tooltips, countries below 15 responses suppressed on maps, heatmap
cells below 10 respondents left blank rather than coloured.

---

## Findings

| # | Finding | Evidence |
| --- | --- | --- |
| 1 | The largest single answer to several policy questions is *"I don't know"* — 45% can't describe the leave policy, 65% don't know if support use is anonymous | Strong |
| 2 | Knowing what's covered matters more than whether it's covered: care-option awareness outranks the existence of benefits in both effect size and model importance | Strong |
| 3 | Mental health is still the riskier disclosure — 23% expect consequences vs 5% for physical health, same question, same person | Strong |
| 4 | Support and treatment rise together in a dose-like pattern, from 37% at support score 0 to 66% at 4–5 | Moderate |
| 5 | The strongest predictor isn't an employer variable at all — family history, 74% vs 35% | Strong |
| 6 | Employer support doesn't close the gender gap; it lifts both groups in parallel | Moderate |

**Model performance:** random forest 0.755 AUC held out (0.773 ± 0.033
cross-validated), logistic regression 0.755. Nearly identical, which says the
structure here is mostly linear. Roughly three-quarters of the variation is not
explained by anything the survey asked.

---

## Limits

- **Self-selected sample.** People who answer a mental health survey are not a
  random draw from the industry. Prevalence figures from this data are biased
  and the direction is not knowable from the data itself.
- **Cross-sectional.** Nothing here can establish that support causes treatment
  rather than the reverse. Supportive employers may attract open people rather
  than produce them.
- **79% male, median age 31, 60% American.**
- **Collected in 2014.** Treat the structure of the relationships as the
  finding; the absolute levels are historical.
- The predictor on the modelling page estimates *survey answers*. It is not a
  clinical instrument and should never be used as one.

---

Data: Open Sourcing Mental Illness (OSMI) Mental Health in Tech Survey 2014,
CC BY-SA 4.0.
