"""Statistical analysis (Phase 5): hypothesis tests + effect sizes.

Research hypotheses
-------------------
H1  Repayment status is associated with default            -> chi-square + Cramér's V
H2  Credit limit differs between default / non-default     -> Welch t-test + Mann-Whitney U
H3  Age differs between default / non-default groups       -> Welch t-test + Mann-Whitney U
+    Demographic associations (SEX / EDUCATION / MARRIAGE) -> chi-square + Cramér's V

Effect sizes are reported next to every p-value (Cramér's V, Cohen's d,
rank-biserial correlation) plus 95% confidence intervals — p-values alone
are never treated as findings.

Usage:
    python -m src.statistical_analysis
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.data_loader import (
    EDUCATION_LABELS,
    MARRIAGE_LABELS,
    PAY_MONTHS,
    SEX_LABELS,
    TARGET,
    load_clean,
)
from src.utils import REPORTS_DIR, save_fig, set_style

PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
ALPHA = 0.05


# --------------------------------------------------------------------------- #
# Effect-size helpers
# --------------------------------------------------------------------------- #
def cramers_v(contingency: pd.DataFrame) -> float:
    """Cramér's V for a contingency table (bias-corrected not needed at n=30k)."""
    chi2 = stats.chi2_contingency(contingency)[0]
    n = contingency.to_numpy().sum()
    r, c = contingency.shape
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1))))


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Pooled-SD Cohen's d."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation for Mann-Whitney U."""
    return float(2 * u_stat / (n1 * n2) - 1)


def mean_diff_ci(a: pd.Series, b: pd.Series) -> tuple[float, float, float]:
    """ Welch mean difference with 95% CI."""
    n1, n2 = len(a), len(b)
    se = np.sqrt(a.var(ddof=1) / n1 + b.var(ddof=1) / n2)
    diff = a.mean() - b.mean()
    dof = (a.var(ddof=1) / n1 + b.var(ddof=1) / n2) ** 2 / (
        (a.var(ddof=1) / n1) ** 2 / (n1 - 1) + (b.var(ddof=1) / n2) ** 2 / (n2 - 1)
    )
    t_crit = stats.t.ppf(0.975, dof)
    return float(diff), float(diff - t_crit * se), float(diff + t_crit * se)


def interpret_effect(v: float, kind: str) -> str:
    """Rough magnitude labels (Cohen's conventions)."""
    if kind == "cramers_v":
        labels = [(0.1, "negligible"), (0.3, "small"), (0.5, "medium")]
    else:  # cohens_d / rank-biserial
        labels = [(0.1, "negligible"), (0.2 / 2, "very small"), (0.35, "small"), (0.65, "medium")]
    for threshold, name in labels:
        if abs(v) < threshold:
            return name
    return "large"


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def chi_square_test(df: pd.DataFrame, col: str, label: str) -> dict:
    table = pd.crosstab(df[col], df[TARGET])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    v = cramers_v(table)
    return {
        "hypothesis": f"{label} vs default",
        "test": "Chi-square test of independence",
        "statistic": round(chi2, 2),
        "p_value": p,
        "effect_size": round(v, 4),
        "effect_type": "Cramér's V",
        "effect_magnitude": interpret_effect(v, "cramers_v"),
        "significant": p < ALPHA,
    }


def two_group_test(df: pd.DataFrame, col: str, label: str) -> dict:
    a = df.loc[df[TARGET] == 1, col]  # default
    b = df.loc[df[TARGET] == 0, col]  # no default

    t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = cohens_d(a, b)
    rbc = rank_biserial(u_stat, len(a), len(b))
    diff, lo, hi = mean_diff_ci(a, b)

    return {
        "hypothesis": f"{label} differs by default group",
        "test": "Welch t-test + Mann-Whitney U",
        "statistic": round(t_stat, 2),
        "p_value": min(t_p, u_p),
        "effect_size": round(d, 4),
        "effect_type": "Cohen's d (rank-biserial "
                       f"{rbc:.3f})",
        "effect_magnitude": interpret_effect(d, "d"),
        "mean_default": round(a.mean(), 2),
        "mean_no_default": round(b.mean(), 2),
        "mean_diff_95ci": f"{diff:.2f} [{lo:.2f}, {hi:.2f}]",
        "significant": min(t_p, u_p) < ALPHA,
    }


def run_all() -> pd.DataFrame:
    set_style()
    df = load_clean()
    results: list[dict] = []

    print("[stats] H1 — repayment status vs default (chi-square) ...")
    for col in PAY_STATUS_COLS:
        res = chi_square_test(df, col, f"Repayment status {PAY_MONTHS[col]} ({col})")
        results.append(res)

    print("[stats] H2 — credit limit between groups ...")
    results.append(two_group_test(df, "LIMIT_BAL", "Credit limit (LIMIT_BAL)"))
    print("[stats] H3 — age between groups ...")
    results.append(two_group_test(df, "AGE", "Age"))

    print("[stats] Demographics ...")
    results.append(chi_square_test(df, "SEX", "Gender"))
    results.append(chi_square_test(df, "EDUCATION", "Education"))
    results.append(chi_square_test(df, "MARRIAGE", "Marital status"))

    out = pd.DataFrame(results)
    out.to_csv(REPORTS_DIR / "statistical_tests.csv", index=False)
    print(out[["hypothesis", "test", "p_value", "effect_size", "effect_type"]]
          .to_string(index=False))

    _fig_effect_sizes(out, df)
    _fig_group_means(df)

    highlights = {
        "pay0_cramers_v": float(out.loc[out.hypothesis.str.contains("PAY_0"), "effect_size"].iloc[0]),
        "limit_cohens_d": float(out.loc[out.hypothesis.str.contains("Credit limit"), "effect_size"].iloc[0]),
        "age_cohens_d": float(out.loc[out.hypothesis.str.contains("Age"), "effect_size"].iloc[0]),
    }
    (REPORTS_DIR / "stats_highlights.json").write_text(json.dumps(highlights, indent=2))
    return out


def _fig_effect_sizes(out: pd.DataFrame, df: pd.DataFrame) -> None:
    pay_rows = out[out.hypothesis.str.contains("Repayment status")]
    months = [PAY_MONTHS[c] for c in PAY_STATUS_COLS]
    # Reverse so the plot reads Apr -> Sep left to right.
    months, vs = months[::-1], pay_rows.effect_size.tolist()[::-1]

    demo = out[out.hypothesis.str.contains("Gender|Education|Marital")]
    demo_names = ["Gender", "Education", "Marital status"]
    demo_vs = demo.effect_size.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    sns.barplot(x=months, y=vs, hue=months, legend=False, palette="Reds", ax=axes[0])
    for i, v in enumerate(vs):
        axes[0].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    axes[0].set_title("H1 — association strength (Cramér's V)\nrepayment status vs default, by month")
    axes[0].set_xlabel("Repayment status month")
    axes[0].set_ylabel("Cramér's V")
    axes[0].set_ylim(0, max(vs) * 1.25)

    sns.barplot(x=demo_names, y=demo_vs, hue=demo_names, legend=False, palette="Greens", ax=axes[1])
    for i, v in enumerate(demo_vs):
        axes[1].text(i, v + 0.004, f"{v:.2f}", ha="center", fontsize=9)
    axes[1].set_title("Demographics — Cramér's V vs default")
    axes[1].set_ylabel("Cramér's V")
    axes[1].set_ylim(0, max(demo_vs) * 1.3 + 0.02)
    save_fig(fig, "14_stat_effect_sizes")


def _fig_group_means(df: pd.DataFrame) -> None:
    specs = [("LIMIT_BAL", "Credit limit (NT$)"), ("AGE", "Age (years)"),
             ("BILL_AMT1", "Latest bill (NT$)"), ("PAY_AMT1", "Latest payment (NT$)")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), constrained_layout=True)
    for ax, (col, label) in zip(axes, specs):
        grp = df.groupby(TARGET)[col].agg(["mean", "sem"]).reset_index()
        grp["ci"] = 1.96 * grp["sem"]
        colors = {0: "#2A9D8F", 1: "#E76F51"}
        ax.bar(["No default", "Default"], grp["mean"], yerr=grp["ci"], capsize=6,
               color=[colors[int(t)] for t in grp[TARGET]])
        for i, row in grp.iterrows():
            ax.text(i, row["mean"] * 1.02, f"{row['mean']:,.0f}", ha="center", fontsize=9)
        ax.set_title(label)
        ax.set_ylabel(label)
    fig.suptitle("Group means with 95% CI — default vs no default", fontsize=13)
    save_fig(fig, "15_stat_group_means")


if __name__ == "__main__":
    run_all()
