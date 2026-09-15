"""Findings: what the analysis concluded, ranked by how much it can support."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import charts as X
from dashboard import data_loader as DL
from dashboard import theme as T
from src import config as C


def render(df: pd.DataFrame, full: pd.DataFrame) -> None:
    T.page_header(
        "Findings", "Six conclusions and the evidence behind each",
        "Ordered by how much weight the data can carry, not by how interesting they sound. Each "
        "one names the chart it came from and the reason it might be wrong.",
        T.ICON["insights"])

    d = full  # findings describe the full dataset, not the filtered view

    care_yes = d[d["care_options"] == "Yes"]["treatment_flag"].mean() * 100
    care_no = d[d["care_options"] == "No"]["treatment_flag"].mean() * 100
    care_ns = d[d["care_options"] == "Not sure"]["treatment_flag"].mean() * 100
    s0 = d[d["support_score"] == 0]["treatment_flag"].mean() * 100
    s45 = d[d["support_score"] >= 4]["treatment_flag"].mean() * 100
    fam_yes = d[d["family_history"] == "Yes"]["treatment_flag"].mean() * 100
    fam_no = d[d["family_history"] == "No"]["treatment_flag"].mean() * 100
    mh = (d["mental_health_consequence"] == "Yes").mean() * 100
    ph = (d["phys_health_consequence"] == "Yes").mean() * 100
    dk_leave = (d["leave"] == "Don't know").mean() * 100
    dk_anon = (d["anonymity"] == "Don't know").mean() * 100
    dk_parity = (d["mental_vs_physical"] == "Don't know").mean() * 100
    female = d[d["gender"] == "Female"]["treatment_flag"].mean() * 100
    male = d[d["gender"] == "Male"]["treatment_flag"].mean() * 100

    findings = [
        {
            "n": 1,
            "title": "The largest answer to several policy questions is “I don’t know”",
            "strength": "Strong",
            "body": (
                f"{dk_leave:.0f}% cannot say how hard it is to take mental health leave. "
                f"{dk_anon:.0f}% cannot say whether using employer resources would stay private. "
                f"{dk_parity:.0f}% cannot say whether their employer treats mental and physical "
                "health equally. These people are not reporting bad policy — they are reporting no "
                "visible policy, and they behave much like the people who report an actively "
                "hostile one."),
            "evidence": "Employer support → Medical leave; Stigma and openness → parity test",
            "caveat": "Self-reported awareness, not an audit of what the employer actually offers. "
                      "Some of these employers may have good policies that nobody has read.",
        },
        {
            "n": 2,
            "title": "Knowing what is covered matters more than whether it is covered",
            "strength": "Strong",
            "body": (
                f"{care_yes:.0f}% of people who know their care options have sought treatment, "
                f"against {care_no:.0f}% of those whose employer does not cover it — and "
                f"{care_ns:.0f}% of the 'not sure' group, which sits between the two. Awareness of "
                "care options carries a larger effect size than the existence of benefits."),
            "evidence": "Treatment drivers → ranked associations; Modelling → permutation importance",
            "caveat": "People already seeking treatment have an obvious reason to have looked up "
                      "their coverage. The arrow may point the other way.",
        },
        {
            "n": 3,
            "title": "Mental health is still the riskier thing to disclose",
            "strength": "Strong",
            "body": (
                f"{mh:.0f}% expect negative consequences from raising mental health with an "
                f"employer, against {ph:.0f}% for physical health — the same question, the same "
                "respondent, the same employer, differing by one word. The interview version of "
                "the question shows the same gap, wider."),
            "evidence": "Stigma and openness → the parity test",
            "caveat": "Measures expectation, not what would actually happen. It is still the "
                      "expectation that governs whether anyone speaks up.",
        },
        {
            "n": 4,
            "title": "Support and treatment rise together, in a dose-like pattern",
            "strength": "Moderate",
            "body": (
                f"Treatment rate climbs from {s0:.0f}% among people who can confirm no employer "
                f"supports to {s45:.0f}% among those who can confirm four or five, rising at "
                "nearly every step rather than jumping once. Openness rises across the same range "
                "while expected penalty falls."),
            "evidence": "Employer support → the support score",
            "caveat": "Cross-sectional data cannot establish direction. Supportive employers may "
                      "attract open people rather than produce them, and industries that hire "
                      "differently may differ in both at once.",
        },
        {
            "n": 5,
            "title": "The strongest single predictor is not an employer variable at all",
            "strength": "Strong",
            "body": (
                f"Family history of mental illness separates the sample more sharply than anything "
                f"else the survey asked: {fam_yes:.0f}% versus {fam_no:.0f}%. In the random forest "
                "it accounts for several times the importance of the next feature. Worth knowing "
                "precisely so that employer factors are judged against a realistic ceiling."),
            "evidence": "Modelling → permutation importance; Treatment drivers → effect sizes",
            "caveat": "Combines genuine heritability with growing up in a household where "
                      "treatment was normal. The survey cannot separate the two.",
        },
        {
            "n": 6,
            "title": "Being well-supported does not close the gender gap",
            "strength": "Moderate",
            "body": (
                f"Women in this sample sought treatment at {female:.0f}% against {male:.0f}% for "
                "men, and the gap persists at every level of employer support. Whatever support "
                "does, it lifts both groups roughly in parallel rather than bringing them together."),
            "evidence": "Respondents → age and gender; Respondents → employment setting",
            "caveat": "251 women against 995 men. The direction is solid; the size of the gap at "
                      "any single support level rests on smaller cells.",
        },
    ]

    for f in findings:
        tone = {"Strong": ("#E3F0EF", C.COLORS["teal"]),
                "Moderate": ("#FBF1DC", "#8A5E0F")}[f["strength"]]
        st.markdown(f"""
        <div class="card" style="margin-bottom:1rem">
          <div style="display:flex;align-items:baseline;gap:0.75rem;margin-bottom:0.45rem">
            <span style="font-family:Sora;font-size:1.5rem;font-weight:600;color:{C.COLORS['line']}">
              {f['n']}</span>
            <span style="font-family:Sora;font-size:1.06rem;font-weight:600;color:{C.COLORS['ink']}">
              {f['title']}</span>
            <span style="margin-left:auto;font-size:0.73rem;font-weight:600;background:{tone[0]};
              color:{tone[1]};padding:0.14rem 0.5rem;border-radius:6px;white-space:nowrap">
              {f['strength']} evidence</span>
          </div>
          <div style="color:{C.COLORS['ink_soft']};font-size:0.92rem;line-height:1.62;
               margin-bottom:0.7rem">{f['body']}</div>
          <div style="display:flex;gap:1.6rem;flex-wrap:wrap;font-size:0.8rem;
               color:{C.COLORS['slate']};border-top:1px solid {C.COLORS['line']};padding-top:0.6rem">
            <div style="flex:1;min-width:230px"><b style="color:{C.COLORS['ink_soft']}">Where to
              see it</b><br>{f['evidence']}</div>
            <div style="flex:1.4;min-width:260px"><b style="color:{C.COLORS['rose']}">Why it might
              be wrong</b><br>{f['caveat']}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    T.section("If you had one thing to change", "ordered by cost, not by effect size")

    actions = pd.DataFrame([
        {"Action": "Publish the mental health leave policy in plain language",
         "Cost": "Near zero",
         "Why this data supports it": f"{dk_leave:.0f}% cannot describe the policy, and their fear "
                                      "profile resembles that of people who say leave is difficult."},
        {"Action": "State plainly whether use of support resources is anonymous",
         "Cost": "Near zero",
         "Why this data supports it": f"{dk_anon:.0f}% do not know — the single largest uncertainty "
                                      "on any question in the survey."},
        {"Action": "Send the specific list of what the health plan covers, by name",
         "Cost": "Low",
         "Why this data supports it": "Awareness of care options outranks the existence of benefits "
                                      "in both the effect-size ranking and the model."},
        {"Action": "Handle the first disclosure on a team carefully",
         "Cost": "Low, but hard",
         "Why this data supports it": "Having witnessed a coworker penalised is associated with a "
                                      "sharp drop in willingness to talk to a supervisor."},
        {"Action": "Add or extend mental health benefits",
         "Cost": "High",
         "Why this data supports it": "Real but smaller effect than awareness, and the four rows "
                                      "above are prerequisites for anyone using them."},
    ])
    st.dataframe(actions, hide_index=True, **T.FULL)

    T.section("Limits", "read before quoting any number from this dashboard")

    left, right = st.columns(2, gap="large")
    with left:
        T.insight(
            "<b>Self-selected sample.</b> People who answer a mental health survey are not a random "
            "draw from the industry. Prevalence figures from this data will be biased and the "
            "direction of that bias is not knowable from the data itself.<br><br>"
            "<b>Self-reported throughout</b>, including the outcome. 'Have you sought treatment' "
            "is a memory and a self-classification, not a medical record.<br><br>"
            "<b>79% male, median age 31</b>, heavily American. The non-binary group is 13 people "
            "and no rate anywhere in this dashboard is calculated from it.",
            title="What the sample can and cannot represent", tone="caveat")
    with right:
        T.insight(
            "<b>Cross-sectional.</b> Every relationship here is an association measured at one "
            "moment. Nothing in this design can establish that support causes treatment rather "
            "than the reverse, or that a third factor produces both.<br><br>"
            "<b>Collected in 2014.</b> Treat the structure of the relationships as the finding and "
            "the absolute levels as historical. Attitudes to mental health at work have moved since."
            "<br><br><b>Not a clinical instrument.</b> The predictor on the modelling page "
            "estimates survey answers, nothing more.",
            title="What the design can and cannot show", tone="caveat")

    st.markdown("")
    st.markdown(
        f"<div style='background:{C.COLORS['ink']};color:#D8E1E9;border-radius:14px;"
        f"padding:1.15rem 1.35rem;font-size:0.88rem;line-height:1.6'>"
        "<b style='color:#fff;font-family:Sora'>If you are reading this because of your own "
        "situation</b><br>This is a 2014 research dataset and a demonstration of analysis "
        "technique — not advice, and not a source to draw conclusions about yourself from. If any "
        "of it lands close to home, a GP or a therapist is the right place to take it.</div>",
        unsafe_allow_html=True)

    T.footnote(
        "Data: Open Sourcing Mental Illness (OSMI) Mental Health in Tech Survey 2014, CC BY-SA 4.0. "
        "Analysis and dashboard built with pandas, scipy, scikit-learn, Plotly and Streamlit.")
