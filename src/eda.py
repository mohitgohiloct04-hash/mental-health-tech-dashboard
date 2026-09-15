"""
Exploratory data analysis: generates the static figure set and a written
findings report from the cleaned survey.

The figures here are the "lab notebook" version of the analysis — they are what
the Streamlit dashboard was designed against. Each function answers one
question and saves one figure.

Run:  python -m src.eda
"""

from __future__ import annotations

import textwrap
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src import analysis as A
from src import config as C

warnings.filterwarnings("ignore")

TEAL_CMAP = LinearSegmentedColormap.from_list("teal", C.SEQ_TEAL)
DIV_CMAP = LinearSegmentedColormap.from_list("div", C.DIVERGING[::-1])


# --------------------------------------------------------------------------
# Shared styling
# --------------------------------------------------------------------------

def set_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": C.COLORS["surface"],
        "axes.facecolor": C.COLORS["surface"],
        "axes.edgecolor": C.COLORS["line"],
        "axes.labelcolor": C.COLORS["ink_soft"],
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": C.COLORS["ink"],
        "axes.titlepad": 14,
        "grid.color": C.COLORS["line"],
        "grid.linewidth": 0.7,
        "text.color": C.COLORS["ink_soft"],
        "xtick.color": C.COLORS["slate"],
        "ytick.color": C.COLORS["slate"],
        "font.size": 10,
        "legend.frameon": False,
        "figure.dpi": 130,
        "savefig.bbox": "tight",
    })


def save(fig: plt.Figure, name: str) -> str:
    path = C.FIGURES / f"{name}.png"
    fig.savefig(path, dpi=150, facecolor=C.COLORS["surface"])
    plt.close(fig)
    return path.name


def annotate_bars(ax, fmt="{:.0f}%", fontsize=9, offset=1.0):
    for patch in ax.patches:
        height = patch.get_height()
        if np.isnan(height) or height == 0:
            continue
        ax.text(patch.get_x() + patch.get_width() / 2, height + offset, fmt.format(height),
                ha="center", va="bottom", fontsize=fontsize, color=C.COLORS["ink_soft"])


def order_for(col, df):
    return C.CATEGORY_ORDER.get(col, sorted(df[col].dropna().unique().tolist()))


# --------------------------------------------------------------------------
# 1. Data quality
# --------------------------------------------------------------------------

def fig_missingness(raw: pd.DataFrame) -> str:
    miss = (raw.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bars = ax.barh(miss.index[::-1], miss.values[::-1], color=C.COLORS["ochre"], height=0.6)
    bars[-1].set_color(C.COLORS["rose"])
    for i, v in enumerate(miss.values[::-1]):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9, color=C.COLORS["ink_soft"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of responses missing")
    ax.set_title("Only four columns have gaps — and two of them are meaningful")
    ax.grid(axis="y", visible=False)
    return save(fig, "01_missingness")


def fig_age_cleaning(raw: pd.DataFrame, df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    raw_age = raw["Age"].clip(-50, 120)
    axes[0].hist(raw_age, bins=60, color=C.COLORS["rose"], alpha=0.85)
    axes[0].set_title("Raw age field (clipped to -50 to 120 to be plottable)")
    axes[0].set_xlabel("Reported age")
    axes[0].annotate("entries include -1726,\n5, 329 and 99,999,999,999",
                     xy=(0.03, 0.72), xycoords="axes fraction", fontsize=9,
                     color=C.COLORS["rose"])
    sns.histplot(df["Age"].dropna(), bins=30, kde=True, color=C.COLORS["teal"], ax=axes[1])
    axes[1].set_title("After validation: 1,251 usable ages, median 31")
    axes[1].set_xlabel("Age")
    for ax in axes:
        ax.set_ylabel("Respondents")
    fig.suptitle("Cleaning the age field", y=1.04, fontsize=14, fontweight="bold",
                 color=C.COLORS["ink"])
    return save(fig, "02_age_cleaning")


def fig_gender_cleaning(raw: pd.DataFrame, df: pd.DataFrame) -> str:
    top = raw["Gender"].value_counts().head(14)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.5, 1]})
    axes[0].barh(top.index[::-1], top.values[::-1], color=C.COLORS["slate"], height=0.65)
    axes[0].set_title(f"{raw['Gender'].nunique()} raw spellings (top 14 shown)")
    axes[0].set_xlabel("Respondents")
    axes[0].grid(axis="y", visible=False)

    counts = df["gender"].value_counts().reindex(C.CATEGORY_ORDER["gender"])
    axes[1].bar(counts.index, counts.values,
                color=[C.ANSWER_COLORS[g] for g in counts.index], width=0.6)
    annotate_bars(axes[1], fmt="{:.0f}", offset=8)
    axes[1].set_title("Collapsed to 3 analytic groups")
    axes[1].set_ylabel("Respondents")
    axes[1].tick_params(axis="x", rotation=15)
    return save(fig, "03_gender_cleaning")


# --------------------------------------------------------------------------
# 2. Who answered
# --------------------------------------------------------------------------

def fig_age_pyramid(df: pd.DataFrame) -> str:
    sub = df[df["gender"].isin(["Male", "Female"])].dropna(subset=["age_group"])
    pivot = pd.crosstab(sub["age_group"], sub["gender"])
    pivot = pivot.reindex(C.CATEGORY_ORDER["age_group"]).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(pivot.index, -pivot["Male"], color=C.COLORS["sky"], height=0.7, label="Male")
    ax.barh(pivot.index, pivot["Female"], color=C.COLORS["plum"], height=0.7, label="Female")
    for i, (m, f) in enumerate(zip(pivot["Male"], pivot["Female"])):
        ax.text(-m - 12, i, f"{int(m)}", va="center", ha="right", fontsize=9)
        ax.text(f + 6, i, f"{int(f)}", va="center", fontsize=9)
    ax.set_xlim(-pivot["Male"].max() * 1.25, pivot["Female"].max() * 1.6)
    ax.set_xticks([])
    ax.set_title("A young, male-skewed sample: 79% male, median age 31")
    ax.legend(loc="lower right")
    ax.grid(visible=False)
    return save(fig, "04_age_pyramid")


def fig_geography(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    top = df["Country"].value_counts().head(12)
    axes[0].barh(top.index[::-1], top.values[::-1], color=C.COLORS["teal"], height=0.65)
    axes[0].set_title("Responses by country (top 12 of 48)")
    axes[0].set_xlabel("Respondents")
    axes[0].grid(axis="y", visible=False)

    states = df[df["Country"] == "United States"]["state"].value_counts().head(12)
    axes[1].barh(states.index[::-1], states.values[::-1], color=C.COLORS["plum"], height=0.65)
    axes[1].set_title("US responses by state (top 12)")
    axes[1].set_xlabel("Respondents")
    axes[1].grid(axis="y", visible=False)
    return save(fig, "05_geography")


def fig_workplace_profile(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    specs = [("no_employees", "Company size"), ("tech_company", "Tech employer"),
             ("remote_work", "Works remotely 50%+"), ("self_employed", "Self-employed")]
    for ax, (col, title) in zip(axes, specs):
        counts = df[col].value_counts()
        if col in C.CATEGORY_ORDER:
            counts = counts.reindex(C.CATEGORY_ORDER[col])
        colors = [C.ANSWER_COLORS.get(str(i), C.COLORS["teal"]) for i in counts.index]
        if col == "no_employees":
            colors = TEAL_CMAP(np.linspace(0.35, 0.95, len(counts)))
        ax.bar(range(len(counts)), counts.values, color=colors, width=0.65)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=35, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Respondents" if col == "no_employees" else "")
    fig.suptitle("Where these people work", y=1.06, fontsize=14, fontweight="bold",
                 color=C.COLORS["ink"])
    return save(fig, "06_workplace_profile")


def fig_response_flow(df: pd.DataFrame) -> str:
    daily = df.groupby("survey_date").size()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.fill_between(daily.index, daily.values, color=C.COLORS["teal_soft"])
    ax.plot(daily.index, daily.values, color=C.COLORS["teal"], linewidth=1.8)
    peak = daily.idxmax()
    ax.annotate(f"{daily.max()} responses on day one",
                xy=(peak, daily.max()), xytext=(0.25, 0.75), textcoords="axes fraction",
                fontsize=9, color=C.COLORS["ink_soft"],
                arrowprops=dict(arrowstyle="->", color=C.COLORS["slate"]))
    ax.set_title("Responses arrived in a single burst — a snapshot, not a time series")
    ax.set_ylabel("Responses")
    return save(fig, "07_response_flow")


# --------------------------------------------------------------------------
# 3. The outcome: treatment
# --------------------------------------------------------------------------

def fig_treatment_split(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    counts = df["treatment"].value_counts()
    axes[0].pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90,
                colors=[C.COLORS["teal"], C.COLORS["slate"]],
                wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
                textprops=dict(color=C.COLORS["ink"]))
    axes[0].set_title("Have sought treatment")

    wi = df["work_interfere"].value_counts().reindex(
        C.CATEGORY_ORDER["work_interfere"] + ["No condition reported"])
    axes[1].bar(range(len(wi)), wi.values,
                color=[C.ANSWER_COLORS.get(i, C.COLORS["slate"]) for i in wi.index], width=0.65)
    axes[1].set_xticks(range(len(wi)))
    axes[1].set_xticklabels([textwrap.fill(i, 12) for i in wi.index], fontsize=8.5)
    axes[1].set_title("Does a condition interfere with work?")

    fam = A.rate_by_group(df, "family_history")
    axes[2].bar(fam["level"], fam["rate"], color=[C.COLORS["slate"], C.COLORS["rose"]], width=0.55)
    axes[2].errorbar(fam["level"], fam["rate"],
                     yerr=[fam["rate"] - fam["ci_low"], fam["ci_high"] - fam["rate"]],
                     fmt="none", ecolor=C.COLORS["ink"], capsize=4, linewidth=1.2)
    annotate_bars(axes[2], offset=3)
    axes[2].set_ylim(0, 100)
    axes[2].set_title("Treatment rate by family history")
    axes[2].set_ylabel("% who sought treatment")
    return save(fig, "08_treatment_overview")


def fig_treatment_drivers(df: pd.DataFrame) -> str:
    specs = [("gender", "Gender"), ("age_group", "Age band"), ("no_employees", "Company size"),
             ("remote_work", "Works remotely"), ("tech_company", "Tech employer"),
             ("region", "Region")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (col, title) in zip(axes.ravel(), specs):
        rates = A.rate_by_group(df, col, min_n=15)
        if col in C.CATEGORY_ORDER:
            rates = rates.set_index("level").reindex(
                [c for c in C.CATEGORY_ORDER[col] if c in rates["level"].values]).reset_index()
        ax.bar(rates["level"], rates["rate"], color=TEAL_CMAP(np.linspace(0.4, 0.9, len(rates))),
               width=0.6)
        ax.errorbar(rates["level"], rates["rate"],
                    yerr=[rates["rate"] - rates["ci_low"], rates["ci_high"] - rates["rate"]],
                    fmt="none", ecolor=C.COLORS["ink_soft"], capsize=3, linewidth=1)
        ax.axhline(df["treatment_flag"].mean() * 100, color=C.COLORS["rose"],
                   linestyle="--", linewidth=1.2)
        for i, (r, n) in enumerate(zip(rates["rate"], rates["n"])):
            ax.text(i, 4, f"n={n}", ha="center", fontsize=8, color="white")
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    fig.suptitle("Treatment rate by respondent group (dashed line = 50.6% overall, bars show 95% CI)",
                 y=1.02, fontsize=13, fontweight="bold", color=C.COLORS["ink"])
    fig.tight_layout()
    return save(fig, "09_treatment_drivers")


def fig_interference_vs_treatment(df: pd.DataFrame) -> str:
    sub = df[df["work_interfere"] != "No condition reported"]
    pivot = pd.crosstab(sub["work_interfere"], sub["treatment"], normalize="index") * 100
    pivot = pivot.reindex(C.CATEGORY_ORDER["work_interfere"])
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(pivot))
    for answer, color in [("Yes", C.COLORS["teal"]), ("No", C.COLORS["slate"])]:
        ax.bar(pivot.index, pivot[answer], bottom=bottom, color=color, width=0.62,
               label=f"Treatment: {answer}")
        for i, v in enumerate(pivot[answer]):
            if v > 6:
                ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")
        bottom += pivot[answer].values
    ax.set_ylim(0, 100)
    ax.set_title("The more a condition interferes with work, the more likely treatment is")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_ylabel("Share of respondents")
    return save(fig, "10_interference_treatment")


# --------------------------------------------------------------------------
# 4. Employer support
# --------------------------------------------------------------------------

def fig_support_matrix(df: pd.DataFrame) -> str:
    rows = []
    for col in ["benefits", "care_options", "wellness_program", "seek_help", "anonymity"]:
        counts = df[col].value_counts(normalize=True) * 100
        rows.append({"question": C.QUESTION_TEXT[col],
                     "Yes": counts.get("Yes", 0),
                     "Not sure / don't know": counts.get("Don't know", 0) + counts.get("Not sure", 0),
                     "No": counts.get("No", 0)})
    mat = pd.DataFrame(rows).set_index("question")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    left = np.zeros(len(mat))
    palette = {"Yes": C.COLORS["teal"], "Not sure / don't know": C.COLORS["ochre"],
               "No": C.COLORS["rose"]}
    labels = [textwrap.fill(i, 46) for i in mat.index]
    for answer, color in palette.items():
        ax.barh(labels, mat[answer], left=left, color=color, height=0.6, label=answer)
        for i, v in enumerate(mat[answer]):
            if v > 7:
                ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
        left += mat[answer].values
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.set_title("What employers actually provide — and how much of it employees can confirm")
    ax.grid(axis="y", visible=False)
    return save(fig, "11_support_matrix")


def fig_support_score(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = df["support_score"].value_counts().sort_index()
    axes[0].bar(counts.index, counts.values, color=TEAL_CMAP(np.linspace(0.25, 0.95, len(counts))),
                width=0.68)
    annotate_bars(axes[0], fmt="{:.0f}", offset=6)
    axes[0].set_title("Support score distribution")
    axes[0].set_xlabel("Number of supports the employee can confirm (0-5)")
    axes[0].set_ylabel("Respondents")
    axes[0].annotate("41% can confirm none", xy=(0.35, 0.85), xycoords="axes fraction",
                     fontsize=10, color=C.COLORS["rose"], fontweight="bold")

    rates = A.rate_by_group(df, "support_score")
    axes[1].plot(rates["level"], rates["rate"], marker="o", color=C.COLORS["teal"], linewidth=2.2)
    axes[1].fill_between(rates["level"], rates["ci_low"], rates["ci_high"],
                         color=C.COLORS["teal_soft"], alpha=0.55)
    axes[1].set_ylim(30, 85)
    axes[1].set_title("Treatment rate rises with employer support")
    axes[1].set_xlabel("Support score")
    axes[1].set_ylabel("% who sought treatment")
    return save(fig, "12_support_score")


def fig_support_by_size(df: pd.DataFrame) -> str:
    pivot = df.pivot_table(index="no_employees", columns="tech_company", values="support_score",
                           aggfunc="mean").reindex(C.CATEGORY_ORDER["no_employees"])
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(pivot))
    ax.bar(x - 0.2, pivot["Yes"], width=0.38, color=C.COLORS["teal"], label="Tech employer")
    ax.bar(x + 0.2, pivot["No"], width=0.38, color=C.COLORS["plum"], label="Non-tech employer")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel("Mean support score (0-5)")
    ax.set_title("Bigger employers provide more — and tech lags non-tech at almost every size")
    ax.legend()
    return save(fig, "13_support_by_size")


def fig_leave(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = df["leave"].value_counts().reindex(C.CATEGORY_ORDER["leave"])
    axes[0].bar(range(len(counts)), counts.values,
                color=[C.ANSWER_COLORS[i] for i in counts.index], width=0.65)
    axes[0].set_xticks(range(len(counts)))
    axes[0].set_xticklabels([textwrap.fill(i, 11) for i in counts.index], fontsize=9)
    axes[0].set_title("How easy is medical leave for mental health?")
    axes[0].set_ylabel("Respondents")
    axes[0].annotate("45% simply do not know", xy=(0.28, 0.88), xycoords="axes fraction",
                     fontsize=10, color=C.COLORS["ochre"], fontweight="bold")

    sub = df.dropna(subset=["leave_ease"])
    pivot = pd.crosstab(sub["leave"], sub["mental_health_consequence"], normalize="index") * 100
    pivot = pivot.reindex([c for c in C.CATEGORY_ORDER["leave"] if c in pivot.index])
    bottom = np.zeros(len(pivot))
    for answer in ["No", "Maybe", "Yes"]:
        axes[1].bar(pivot.index, pivot[answer], bottom=bottom, width=0.6,
                    color=C.ANSWER_COLORS[answer], label=answer)
        bottom += pivot[answer].values
    axes[1].set_xticklabels([textwrap.fill(i, 11) for i in pivot.index], fontsize=9)
    axes[1].set_title("Fear of consequences, by how easy leave is")
    axes[1].legend(title="Expects consequences", loc="upper center",
                   bbox_to_anchor=(0.5, -0.12), ncol=3)
    axes[1].set_ylim(0, 100)
    return save(fig, "14_leave")


# --------------------------------------------------------------------------
# 5. Stigma and openness
# --------------------------------------------------------------------------

def fig_parity(df: pd.DataFrame) -> str:
    parity = A.parity_summary(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (question, group) in zip(axes, parity.groupby("question", sort=False)):
        x = np.arange(len(group))
        ax.bar(x - 0.2, group["Mental health"], width=0.38, color=C.COLORS["plum"],
               label="Mental health")
        ax.bar(x + 0.2, group["Physical health"], width=0.38, color=C.COLORS["sky"],
               label="Physical health")
        for i, (m, p) in enumerate(zip(group["Mental health"], group["Physical health"])):
            ax.text(i - 0.2, m + 1.5, f"{m:.0f}%", ha="center", fontsize=9)
            ax.text(i + 0.2, p + 1.5, f"{p:.0f}%", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(group["answer"])
        ax.set_ylim(0, 100)
        ax.set_title(textwrap.fill(question, 42), fontsize=11)
        ax.set_ylabel("% of respondents")
        ax.legend()
    fig.suptitle("The same question, asked twice: mental health is treated as the riskier disclosure",
                 y=1.04, fontsize=13, fontweight="bold", color=C.COLORS["ink"])
    return save(fig, "15_parity_gap")


def fig_openness(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, col in zip(axes[:2], ["coworkers", "supervisor"]):
        counts = df[col].value_counts().reindex(C.CATEGORY_ORDER[col])
        ax.bar(counts.index, counts.values,
               color=[C.ANSWER_COLORS[i] for i in counts.index], width=0.6)
        annotate_bars(ax, fmt="{:.0f}", offset=6)
        ax.set_title(textwrap.fill(C.QUESTION_TEXT[col], 38), fontsize=10.5)
        ax.set_ylabel("Respondents")

    counts = df["openness_score"].value_counts().sort_index()
    axes[2].bar(counts.index, counts.values,
                color=TEAL_CMAP(np.linspace(0.25, 0.95, len(counts))), width=0.68)
    axes[2].set_title("Openness score (0-4)", fontsize=10.5)
    axes[2].set_xlabel("Willingness to discuss with coworkers + supervisor")
    return save(fig, "16_openness")


def fig_stigma_vs_support(df: pd.DataFrame) -> str:
    pivot = df.pivot_table(index="support_score", columns="obs_consequence",
                           values="stigma_score", aggfunc="mean")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=DIV_CMAP, ax=axes[0],
                cbar_kws={"label": "Mean stigma score"}, linewidths=1.2, linecolor="white")
    axes[0].set_title("Stigma score by support level and witnessed consequences")
    axes[0].set_xlabel("Has seen a coworker penalised")
    axes[0].set_ylabel("Support score")

    grouped = df.groupby("support_score")[["openness_score", "stigma_score"]].mean()
    axes[1].plot(grouped.index, grouped["openness_score"], marker="o", linewidth=2.2,
                 color=C.COLORS["teal"], label="Openness score (0-4)")
    axes[1].plot(grouped.index, grouped["stigma_score"], marker="s", linewidth=2.2,
                 color=C.COLORS["rose"], label="Stigma score (0-5)")
    axes[1].set_xlabel("Support score")
    axes[1].set_title("More support, more openness, less expected penalty")
    axes[1].legend()
    return save(fig, "17_stigma_support")


def fig_consequence_matrix(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    pivot = pd.crosstab(df["mental_health_consequence"], df["supervisor"])
    pivot = pivot.reindex(index=["No", "Maybe", "Yes"], columns=C.CATEGORY_ORDER["supervisor"])
    sns.heatmap(pivot, annot=True, fmt="d", cmap=TEAL_CMAP, linewidths=1.4, linecolor="white",
                cbar_kws={"label": "Respondents"}, ax=ax)
    ax.set_xlabel("Would discuss with supervisor")
    ax.set_ylabel("Expects negative consequences")
    ax.set_title("Fear of consequences maps almost directly onto silence")
    return save(fig, "18_consequence_matrix")


# --------------------------------------------------------------------------
# 6. Multivariate
# --------------------------------------------------------------------------

def fig_correlation(df: pd.DataFrame) -> str:
    corr = A.index_correlations(df)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=DIV_CMAP, center=0,
                vmin=-0.6, vmax=0.6, linewidths=1.2, linecolor="white",
                cbar_kws={"label": "Spearman rho"}, ax=ax, annot_kws={"fontsize": 9})
    ax.set_title("Spearman correlation between the engineered indices")
    return save(fig, "19_correlation")


def fig_association_strength(assoc: pd.DataFrame) -> str:
    top = assoc.head(15).iloc[::-1]
    colors = [C.COLORS["slate"] if leaky else C.COLORS["teal"] for leaky in top["leaky"]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh([textwrap.fill(q, 52) for q in top["question"]], top["cramers_v"],
            color=colors, height=0.66)
    for i, (v, sig) in enumerate(zip(top["cramers_v"], top["significant"])):
        ax.text(v + 0.008, i, f"{v:.2f}{'' if sig else ' (n.s.)'}", va="center", fontsize=9)
    for threshold, label in [(0.1, "small"), (0.2, "moderate"), (0.35, "strong")]:
        ax.axvline(threshold, color=C.COLORS["line"], linestyle="--", linewidth=1)
        ax.text(threshold, len(top) - 0.3, label, fontsize=8, color=C.COLORS["slate"], ha="center")
    ax.set_xlabel("Cramer's V (effect size)")
    ax.set_title("Strength of association with having sought treatment\n"
                 "Grey = work interference, which is a symptom rather than a cause",
                 fontsize=12)
    ax.grid(axis="y", visible=False)
    return save(fig, "20_association_strength")


def fig_model_performance(res: dict) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for name, color in [("Logistic regression", C.COLORS["teal"]),
                        ("Random forest", C.COLORS["plum"])]:
        roc = res[name]["roc"]
        axes[0].plot(roc["fpr"], roc["tpr"], color=color, linewidth=2.2,
                     label=f"{name} (AUC {res[name]['roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color=C.COLORS["slate"], linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curves on held-out data")
    axes[0].legend(loc="lower right", fontsize=9)

    cm = np.array(res["Random forest"]["confusion"])
    sns.heatmap(cm, annot=True, fmt="d", cmap=TEAL_CMAP, cbar=False, linewidths=1.5,
                linecolor="white", ax=axes[1],
                xticklabels=["Predicted no", "Predicted yes"],
                yticklabels=["Actual no", "Actual yes"])
    axes[1].set_title(f"Random forest confusion matrix (n={res['test_size']})")

    imp = res["importance"].head(10).iloc[::-1]
    axes[2].barh(imp["feature"], imp["importance"], xerr=imp["std"],
                 color=C.COLORS["teal"], height=0.65,
                 error_kw=dict(ecolor=C.COLORS["ink_soft"], capsize=3, linewidth=1))
    axes[2].set_xlabel("Drop in AUC when shuffled")
    axes[2].set_title("Permutation importance")
    axes[2].grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, "21_model_performance")


def fig_odds_ratios(res: dict) -> str:
    odds = res["odds_ratios"].head(16).iloc[::-1]
    colors = [C.COLORS["teal"] if o >= 1 else C.COLORS["rose"] for o in odds["odds_ratio"]]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.barh(odds["term"], odds["odds_ratio"] - 1, left=1, color=colors, height=0.65)
    ax.axvline(1, color=C.COLORS["ink"], linewidth=1.2)
    for i, o in enumerate(odds["odds_ratio"]):
        ax.text(o + (0.02 if o >= 1 else -0.02), i, f"{o:.2f}x", va="center",
                ha="left" if o >= 1 else "right", fontsize=9)
    ax.set_xlabel("Odds ratio for having sought treatment (1.0 = no effect)")
    ax.set_title("Logistic regression: what moves the odds, holding everything else constant")
    ax.grid(axis="y", visible=False)
    return save(fig, "22_odds_ratios")


def fig_segment_heatmap(df: pd.DataFrame) -> str:
    pivot = df.pivot_table(index="no_employees", columns="age_group",
                           values="treatment_flag", aggfunc="mean", observed=True) * 100
    pivot = pivot.reindex(index=C.CATEGORY_ORDER["no_employees"],
                          columns=C.CATEGORY_ORDER["age_group"])
    counts = df.pivot_table(index="no_employees", columns="age_group", values="treatment_flag",
                            aggfunc="count", observed=True).reindex(
        index=C.CATEGORY_ORDER["no_employees"], columns=C.CATEGORY_ORDER["age_group"])
    labels = pivot.round(0).astype("Int64").astype(str) + "%\nn=" + counts.astype("Int64").astype(str)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    sns.heatmap(pivot, annot=labels.values, fmt="", cmap=TEAL_CMAP, linewidths=1.4,
                linecolor="white", cbar_kws={"label": "% who sought treatment"}, ax=ax,
                annot_kws={"fontsize": 8.5})
    ax.set_title("Treatment rate by company size and age band")
    ax.set_xlabel("Age band")
    ax.set_ylabel("Company size")
    return save(fig, "23_segment_heatmap")


def fig_gender_support_interaction(df: pd.DataFrame) -> str:
    sub = df[df["gender"].isin(["Male", "Female"])]
    pivot = sub.pivot_table(index="support_band", columns="gender", values="treatment_flag",
                            aggfunc="mean", observed=True) * 100
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(pivot))
    ax.bar(x - 0.2, pivot["Male"], width=0.38, color=C.COLORS["sky"], label="Male")
    ax.bar(x + 0.2, pivot["Female"], width=0.38, color=C.COLORS["plum"], label="Female")
    for i, (m, f) in enumerate(zip(pivot["Male"], pivot["Female"])):
        ax.text(i - 0.2, m + 1.5, f"{m:.0f}%", ha="center", fontsize=9)
        ax.text(i + 0.2, f + 1.5, f"{f:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% who sought treatment")
    ax.set_title("Support helps both groups, but the gender gap never closes")
    ax.legend()
    return save(fig, "24_gender_support")


# --------------------------------------------------------------------------
# Findings report
# --------------------------------------------------------------------------

def write_report(df: pd.DataFrame, assoc: pd.DataFrame, res: dict, figures: list[str]) -> None:
    rate = df["treatment_flag"].mean() * 100
    fam_yes = df[df["family_history"] == "Yes"]["treatment_flag"].mean() * 100
    fam_no = df[df["family_history"] == "No"]["treatment_flag"].mean() * 100
    care_yes = df[df["care_options"] == "Yes"]["treatment_flag"].mean() * 100
    care_no = df[df["care_options"] == "No"]["treatment_flag"].mean() * 100
    support0 = df[df["support_score"] == 0]["treatment_flag"].mean() * 100
    support5 = df[df["support_score"] >= 4]["treatment_flag"].mean() * 100
    female = df[df["gender"] == "Female"]["treatment_flag"].mean() * 100
    male = df[df["gender"] == "Male"]["treatment_flag"].mean() * 100
    dk_leave = (df["leave"] == "Don't know").mean() * 100
    mh_int = (df["mental_health_interview"] == "No").mean() * 100
    ph_int = (df["phys_health_interview"] == "No").mean() * 100
    mh_cons = (df["mental_health_consequence"] == "Yes").mean() * 100
    ph_cons = (df["phys_health_consequence"] == "Yes").mean() * 100
    parity_dk = (df["mental_vs_physical"] == "Don't know").mean() * 100

    lines = [
        "# Mental Health in Tech — EDA findings",
        "",
        f"Base: {len(df):,} responses to the 2014 OSMI survey, {df['Country'].nunique()} countries. "
        "Self-selected sample, collected in a single burst, so it describes the people who chose "
        "to answer rather than the industry as a whole.",
        "",
        "## Headline",
        "",
        f"- **{rate:.1f}% have sought treatment** for a mental health condition — an almost even split.",
        f"- **Family history is the single strongest non-symptomatic predictor**: {fam_yes:.0f}% vs "
        f"{fam_no:.0f}% treatment rate (Cramer's V = "
        f"{assoc.loc[assoc.feature == 'family_history', 'cramers_v'].iloc[0]:.2f}).",
        f"- **Knowing your care options is worth more than having them**: {care_yes:.0f}% of people who "
        f"know what their employer covers have sought treatment, against {care_no:.0f}% of those who "
        "say it is not covered — and the 'Don't know' group sits in between, which is the gap policy "
        "communication can actually close.",
        f"- **Support compounds**: treatment rate climbs from {support0:.0f}% where an employee can "
        f"confirm no supports to {support5:.0f}% where they can confirm four or five.",
        f"- **Mental health is still the riskier disclosure**: {mh_cons:.0f}% expect negative "
        f"consequences from raising mental health with an employer against {ph_cons:.0f}% for physical "
        f"health; {mh_int:.0f}% would not mention it in an interview against {ph_int:.0f}% for physical "
        "health.",
        "",
        "## What the models say",
        "",
        f"Excluding work interference (a symptom of the outcome rather than a cause), a random forest "
        f"reaches **{res['Random forest']['roc_auc']:.3f} AUC** on held-out data "
        f"({res['Random forest']['cv_auc_mean']:.3f} +/- {res['Random forest']['cv_auc_std']:.3f} in "
        f"5-fold cross-validation) and logistic regression reaches "
        f"{res['Logistic regression']['roc_auc']:.3f}. That is a real signal but far from "
        "deterministic — roughly a quarter of the variation is explained, and the rest is personal "
        "circumstance the survey never asked about.",
        "",
        "Permutation importance ranks the drivers as:",
        "",
    ]
    for _, row in res["importance"].head(6).iterrows():
        lines.append(f"{len(lines) and ''}- `{row['feature']}` — {row['importance']:.3f} AUC drop when shuffled")

    lines += [
        "",
        "## The blind-spot finding",
        "",
        f"The largest single answer to several policy questions is *\"I don't know\"*: {dk_leave:.0f}% "
        f"cannot say how hard it is to take mental health leave, and {parity_dk:.0f}% cannot say whether "
        "their employer takes mental health as seriously as physical health. These are not people "
        "reporting bad policy — they are people reporting no visible policy at all. For an employer, "
        "that is the cheapest group to move.",
        "",
        "## Caveats worth carrying into any use of this",
        "",
        "- The sample is 79% male and median age 31; the non-binary group is 13 people and no rate "
        "should be read off it.",
        "- Every measure is self-reported, including the outcome.",
        "- The data is from 2014. Treat the structure of the relationships as the finding, not the "
        "absolute levels.",
        "- Associations here are not causal. Support and treatment move together; this data cannot "
        "say which direction the arrow points.",
        "",
        "## Figure index",
        "",
    ]
    for name in figures:
        lines.append(f"- `figures/{name}`")

    (C.REPORTS / "eda_findings.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------

def main() -> None:
    set_style()
    raw = pd.read_csv(C.RAW_DATA)
    df = pd.read_csv(C.CLEAN_DATA)
    df["age_group"] = pd.Categorical(df["age_group"], C.CATEGORY_ORDER["age_group"], ordered=True)

    assoc = A.association_table(df)
    res = A.fit_models(df)

    figures = [
        fig_missingness(raw),
        fig_age_cleaning(raw, df),
        fig_gender_cleaning(raw, df),
        fig_age_pyramid(df),
        fig_geography(df),
        fig_workplace_profile(df),
        fig_response_flow(df),
        fig_treatment_split(df),
        fig_treatment_drivers(df),
        fig_interference_vs_treatment(df),
        fig_support_matrix(df),
        fig_support_score(df),
        fig_support_by_size(df),
        fig_leave(df),
        fig_parity(df),
        fig_openness(df),
        fig_stigma_vs_support(df),
        fig_consequence_matrix(df),
        fig_correlation(df),
        fig_association_strength(assoc),
        fig_model_performance(res),
        fig_odds_ratios(res),
        fig_segment_heatmap(df),
        fig_gender_support_interaction(df),
    ]
    write_report(df, assoc, res, figures)
    print(f"Wrote {len(figures)} figures to {C.FIGURES.relative_to(C.ROOT)}")
    print(f"Wrote findings to {(C.REPORTS / 'eda_findings.md').relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
