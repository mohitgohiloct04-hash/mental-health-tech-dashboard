# Mental Health in Tech — EDA findings

Base: 1,259 responses to the 2014 OSMI survey, 48 countries. Self-selected sample, collected in a single burst, so it describes the people who chose to answer rather than the industry as a whole.

## Headline

- **50.6% have sought treatment** for a mental health condition — an almost even split.
- **Family history is the single strongest non-symptomatic predictor**: 74% vs 35% treatment rate (Cramer's V = 0.38).
- **Knowing your care options is worth more than having them**: 69% of people who know what their employer covers have sought treatment, against 41% of those who say it is not covered — and the 'Don't know' group sits in between, which is the gap policy communication can actually close.
- **Support compounds**: treatment rate climbs from 37% where an employee can confirm no supports to 66% where they can confirm four or five.
- **Mental health is still the riskier disclosure**: 23% expect negative consequences from raising mental health with an employer against 5% for physical health; 80% would not mention it in an interview against 40% for physical health.

## What the models say

Excluding work interference (a symptom of the outcome rather than a cause), a random forest reaches **0.755 AUC** on held-out data (0.773 +/- 0.033 in 5-fold cross-validation) and logistic regression reaches 0.755. That is a real signal but far from deterministic — roughly a quarter of the variation is explained, and the rest is personal circumstance the survey never asked about.

Permutation importance ranks the drivers as:

- `family_history` — 0.097 AUC drop when shuffled
- `care_options` — 0.021 AUC drop when shuffled
- `gender` — 0.011 AUC drop when shuffled
- `stigma_score` — 0.008 AUC drop when shuffled
- `obs_consequence` — 0.005 AUC drop when shuffled
- `support_score` — 0.002 AUC drop when shuffled

## The blind-spot finding

The largest single answer to several policy questions is *"I don't know"*: 45% cannot say how hard it is to take mental health leave, and 46% cannot say whether their employer takes mental health as seriously as physical health. These are not people reporting bad policy — they are people reporting no visible policy at all. For an employer, that is the cheapest group to move.

## Caveats worth carrying into any use of this

- The sample is 79% male and median age 31; the non-binary group is 13 people and no rate should be read off it.
- Every measure is self-reported, including the outcome.
- The data is from 2014. Treat the structure of the relationships as the finding, not the absolute levels.
- Associations here are not causal. Support and treatment move together; this data cannot say which direction the arrow points.

## Figure index

- `figures/01_missingness.png`
- `figures/02_age_cleaning.png`
- `figures/03_gender_cleaning.png`
- `figures/04_age_pyramid.png`
- `figures/05_geography.png`
- `figures/06_workplace_profile.png`
- `figures/07_response_flow.png`
- `figures/08_treatment_overview.png`
- `figures/09_treatment_drivers.png`
- `figures/10_interference_treatment.png`
- `figures/11_support_matrix.png`
- `figures/12_support_score.png`
- `figures/13_support_by_size.png`
- `figures/14_leave.png`
- `figures/15_parity_gap.png`
- `figures/16_openness.png`
- `figures/17_stigma_support.png`
- `figures/18_consequence_matrix.png`
- `figures/19_correlation.png`
- `figures/20_association_strength.png`
- `figures/21_model_performance.png`
- `figures/22_odds_ratios.png`
- `figures/23_segment_heatmap.png`
- `figures/24_gender_support.png`