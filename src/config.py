"""
Central configuration for the Mental Health in Tech analytics project.

Everything that more than one module needs to agree on lives here: file
locations, the colour system, how survey answers map onto numbers, and which
columns belong to which conceptual block of the questionnaire.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = ROOT / "data" / "raw" / "survey.csv"
CLEAN_DATA = ROOT / "data" / "processed" / "survey_clean.csv"
CLEANING_LOG = ROOT / "data" / "processed" / "cleaning_log.json"

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

for _p in (CLEAN_DATA.parent, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Design system
# --------------------------------------------------------------------------
# The palette is built around a deep teal (the working colour for "support
# exists") and a warm ochre (the working colour for "support is unclear"),
# with plum reserved for stigma measures. Keeping these three meanings stable
# across every chart means a reader learns the colour language once.

COLORS = {
    "ink": "#131B23",        # headings, sidebar background
    "ink_soft": "#2A3742",   # secondary text on light
    "slate": "#64748B",      # muted labels, gridlines
    "canvas": "#EEF1F4",     # app background
    "surface": "#FFFFFF",    # cards
    "line": "#DCE3E9",       # hairlines, borders
    "teal": "#0F766E",       # primary — support, positive
    "teal_soft": "#B7DEDA",
    "ochre": "#C98B1E",      # ambiguity — "Don't know"
    "ochre_soft": "#F2DDB0",
    "plum": "#6D4B7A",       # stigma, perception
    "plum_soft": "#DCCCE4",
    "rose": "#B23A48",       # risk, negative outcomes
    "rose_soft": "#F0C9CE",
    "sky": "#3E7CB1",        # neutral comparison series
}

# Diverging / sequential ramps used by heatmaps and choropleths.
SEQ_TEAL = ["#F2F7F7", "#CDE5E3", "#9ACBC7", "#5DAAA4", "#2B8A82", "#0F766E"]
SEQ_WARM = ["#FBF6EC", "#F2E2C0", "#E6C889", "#D8AC55", "#C98B1E", "#9C6A12"]
DIVERGING = ["#0F766E", "#5DAAA4", "#B7DEDA", "#EEF1F4", "#F0C9CE", "#D1798A", "#B23A48"]

# Stable colours for the answer values that recur across dozens of questions.
ANSWER_COLORS = {
    "Yes": COLORS["teal"],
    "No": COLORS["rose"],
    "Maybe": COLORS["ochre"],
    "Don't know": COLORS["slate"],
    "Not sure": COLORS["slate"],
    "Some of them": COLORS["ochre"],
    "Never": COLORS["teal"],
    "Rarely": COLORS["teal_soft"],
    "Sometimes": COLORS["ochre"],
    "Often": COLORS["rose"],
    "Very easy": COLORS["teal"],
    "Somewhat easy": COLORS["teal_soft"],
    "Somewhat difficult": COLORS["ochre"],
    "Very difficult": COLORS["rose"],
    "Male": COLORS["sky"],
    "Female": COLORS["plum"],
    "Non-binary / other": COLORS["ochre"],
}

FONT_HEADING = "Sora"
FONT_BODY = "Inter"


# --------------------------------------------------------------------------
# Question blocks
# --------------------------------------------------------------------------
# The 2014 OSMI questionnaire moves through four themes. Grouping the columns
# this way is what lets the dashboard have real sections rather than 26
# unrelated bar charts.

DEMOGRAPHIC_COLS = ["Age", "Gender", "Country", "state", "self_employed", "no_employees",
                    "remote_work", "tech_company"]

CLINICAL_COLS = ["family_history", "treatment", "work_interfere"]

# Does the employer provide something concrete?
SUPPORT_COLS = ["benefits", "care_options", "wellness_program", "seek_help", "anonymity", "leave"]

# Does the respondent expect a penalty for being open?
STIGMA_COLS = ["mental_health_consequence", "phys_health_consequence", "coworkers", "supervisor",
               "mental_health_interview", "phys_health_interview", "mental_vs_physical",
               "obs_consequence"]

# Pairs where the survey asked the same question about mental and physical
# health. The gap between the two is the cleanest stigma measure in the data.
PARITY_PAIRS = [
    ("mental_health_consequence", "phys_health_consequence",
     "Expects negative consequences from discussing it with the employer"),
    ("mental_health_interview", "phys_health_interview",
     "Would bring it up in a job interview"),
]


# --------------------------------------------------------------------------
# Ordinal encodings
# --------------------------------------------------------------------------
# Several questions are ordered, not merely categorical. Encoding them keeps
# bar charts in a sensible order and makes correlation analysis meaningful.

ORDINAL_MAPS = {
    "no_employees": {"1-5": 0, "6-25": 1, "26-100": 2, "100-500": 3,
                     "500-1000": 4, "More than 1000": 5},
    "work_interfere": {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3},
    "leave": {"Very difficult": 0, "Somewhat difficult": 1, "Don't know": None,
              "Somewhat easy": 2, "Very easy": 3},
    "coworkers": {"No": 0, "Some of them": 1, "Yes": 2},
    "supervisor": {"No": 0, "Some of them": 1, "Yes": 2},
    "mental_health_consequence": {"No": 0, "Maybe": 1, "Yes": 2},
    "phys_health_consequence": {"No": 0, "Maybe": 1, "Yes": 2},
    "mental_health_interview": {"No": 0, "Maybe": 1, "Yes": 2},
    "phys_health_interview": {"No": 0, "Maybe": 1, "Yes": 2},
}

CATEGORY_ORDER = {
    "no_employees": ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"],
    "work_interfere": ["Never", "Rarely", "Sometimes", "Often"],
    "leave": ["Very difficult", "Somewhat difficult", "Don't know", "Somewhat easy", "Very easy"],
    "age_group": ["Under 25", "25-29", "30-34", "35-39", "40-49", "50+"],
    "coworkers": ["No", "Some of them", "Yes"],
    "supervisor": ["No", "Some of them", "Yes"],
    "gender": ["Male", "Female", "Non-binary / other"],
    "yes_no_maybe": ["No", "Maybe", "Yes"],
    "yes_no_dk": ["No", "Don't know", "Yes"],
}

# Plain-English question text, used as chart titles and tooltips so the
# dashboard reads like a survey report rather than a database dump.
QUESTION_TEXT = {
    "self_employed": "Are you self-employed?",
    "family_history": "Do you have a family history of mental illness?",
    "treatment": "Have you sought treatment for a mental health condition?",
    "work_interfere": "If you have a condition, does it interfere with your work?",
    "no_employees": "How many employees does your company have?",
    "remote_work": "Do you work remotely at least 50% of the time?",
    "tech_company": "Is your employer primarily a tech company?",
    "benefits": "Does your employer provide mental health benefits?",
    "care_options": "Do you know the mental health care options your employer covers?",
    "wellness_program": "Has your employer discussed mental health as part of wellness?",
    "seek_help": "Does your employer provide resources for seeking help?",
    "anonymity": "Is your anonymity protected if you use these resources?",
    "leave": "How easy is it to take medical leave for a mental health condition?",
    "mental_health_consequence": "Would discussing mental health with your employer have negative consequences?",
    "phys_health_consequence": "Would discussing physical health with your employer have negative consequences?",
    "coworkers": "Would you discuss a mental health issue with your coworkers?",
    "supervisor": "Would you discuss a mental health issue with your supervisor?",
    "mental_health_interview": "Would you raise mental health with a potential employer in an interview?",
    "phys_health_interview": "Would you raise physical health with a potential employer in an interview?",
    "mental_vs_physical": "Does your employer take mental health as seriously as physical health?",
    "obs_consequence": "Have you seen negative consequences for coworkers who were open?",
}

# Engineered columns, described once so the dashboard can explain itself.
DERIVED_TEXT = {
    "support_score": "Count of five concrete employer supports the respondent can confirm "
                     "(benefits, care options, wellness programme, help resources, anonymity). 0-5.",
    "openness_score": "Willingness to discuss a mental health issue with coworkers and with a "
                      "supervisor, summed. 0-4.",
    "stigma_score": "Expected penalty for openness: fear of consequences at work, unwillingness to "
                    "raise it in an interview, and having witnessed consequences. 0-5.",
    "parity_gap": "Mental-health consequence score minus physical-health consequence score. "
                  "Positive means the respondent fears talking about mental health more.",
}

RANDOM_STATE = 42
