"""Exploratory Data Analysis (Phase 4): univariate, bivariate, target analysis.

Produces the figure set used in the README and the Streamlit "EDA" page:

    01_target_distribution       class balance
    02_age_distribution          age vs default
    03_limit_bal_distribution    credit-limit distribution (raw + log view)
    04_bill_amounts              BILL_AMT1..6 histograms
    05_payment_amounts           PAY_AMT1..6 histograms (log1p view)
    06_categorical_default_rates default rate by SEX / EDUCATION / MARRIAGE
    07_repayment_status          PAY_0 behaviour vs default
    08_repayment_heatmap         default rate by month x repayment status
    09_boxplots_by_target        AGE / LIMIT_BAL / BILL_AMT1 / PAY_AMT1
    10_violin_by_target          distribution shapes for AGE / LIMIT_BAL
    11_stacked_proportions       default share within demographic groups
    12_correlation_heatmap       full numeric correlation matrix
    13_limit_decile_default      default rate across credit-limit deciles

Usage:
    python -m src.eda
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loader import (
    EDUCATION_LABELS,
    MARRIAGE_LABELS,
    PAY_MONTHS,
    PAY_SCALE,
    SEX_LABELS,
    TARGET,
    load_clean,
)
from src.utils import ACCENT, COLOR_NO, COLOR_YES, REPORTS_DIR, save_fig, set_style

BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

TARGET_LABELS = {0: "No default", 1: "Default"}


def _label_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable labels for categorical plots."""
    out = df.copy()
    out["SEX_LABEL"] = out["SEX"].map(SEX_LABELS)
    out["EDUCATION_LABEL"] = out["EDUCATION"].map(EDUCATION_LABELS)
    out["MARRIAGE_LABEL"] = out["MARRIAGE"].map(MARRIAGE_LABELS)
    out["TARGET_LABEL"] = out[TARGET].map(TARGET_LABELS)
    return out


# --------------------------------------------------------------------------- #
# A. Univariate + target
# --------------------------------------------------------------------------- #
def fig_target_distribution(df: pd.DataFrame) -> None:
    counts = df[TARGET].value_counts().sort_index()
    rate = df[TARGET].mean()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    sns.countplot(
        data=df, x=TARGET, hue=TARGET, legend=False,
        palette={0: COLOR_NO, 1: COLOR_YES}, ax=axes[0],
    )
    for idx, n in counts.items():
        axes[0].text(idx, n + 300, f"{n:,}\n({n / len(df):.1%})", ha="center", fontsize=10)
    axes[0].set_xticks([0, 1], ["No default (0)", "Default (1)"])
    axes[0].set_ylabel("Clients")
    axes[0].set_title("Target distribution (class balance)")

    axes[1].pie(counts, labels=[f"No default\n{counts[0]:,}", f"Default\n{counts[1]:,}"],
                colors=[COLOR_NO, COLOR_YES], autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(width=0.45), textprops={"fontsize": 10})
    axes[1].set_title(f"Overall default rate: {rate:.1%}")
    save_fig(fig, "01_target_distribution")


def fig_age_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    sns.histplot(data=df, x="AGE", bins=np.arange(21, 80, 2), color=ACCENT, ax=axes[0])
    axes[0].set_title("Age distribution (all clients)")
    axes[0].set_xlabel("Age (years)")

    sns.kdeplot(data=df, x="AGE", hue="TARGET_LABEL", common_norm=False,
                palette={"No default": COLOR_NO, "Default": COLOR_YES},
                fill=True, alpha=0.25, ax=axes[1])
    axes[1].set_title("Age distribution by default outcome")
    axes[1].set_xlabel("Age (years)")
    save_fig(fig, "02_age_distribution")


def fig_limit_bal_distribution(df: pd.DataFrame) -> None:
    p99 = df["LIMIT_BAL"].quantile(0.99)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    sns.histplot(data=df, x="LIMIT_BAL", bins=40, color=ACCENT, ax=axes[0])
    axes[0].set_title(f"Credit limit (NT$) — 99th pct dashed at {p99:,.0f}")
    axes[0].axvline(p99, ls="--", color=COLOR_YES)
    axes[0].set_xlabel("LIMIT_BAL (NT$)")

    sns.kdeplot(data=df, x="LIMIT_BAL", hue="TARGET_LABEL", common_norm=False,
                palette={"No default": COLOR_NO, "Default": COLOR_YES},
                fill=True, alpha=0.25, ax=axes[1])
    axes[1].set_xlim(0, p99)
    axes[1].set_title("Credit limit by default outcome (zoomed)")
    axes[1].set_xlabel("LIMIT_BAL (NT$)")
    save_fig(fig, "03_limit_bal_distribution")


def fig_bill_amounts(df: pd.DataFrame) -> None:
    # BILL_AMT1 shares its month with PAY_0 (Sep); BILL_AMTi -> PAY_i for i >= 2.
    bill_month = {f"BILL_AMT{i}": PAY_MONTHS["PAY_0" if i == 1 else f"PAY_{i}"] for i in range(1, 7)}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True, sharey=True)
    for ax, col in zip(axes.ravel(), BILL_COLS):
        sns.histplot(data=df, x=col, bins=40, color=ACCENT, ax=ax)
        ax.set_title(f"{col} ({bill_month[col]})")
        ax.set_xlabel("NT$")
    fig.suptitle("Monthly bill amounts — BILL_AMT1 (Sep) … BILL_AMT6 (Apr)", y=1.04, fontsize=13)
    save_fig(fig, "04_bill_amounts")


def fig_payment_amounts(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    for ax, col in zip(axes.ravel(), PAYAMT_COLS):
        sns.histplot(data=df, x=np.log1p(df[col].clip(lower=0)), bins=40,
                     color=ACCENT, ax=ax)
        ax.set_title(f"{col} (log1p)")
        ax.set_xlabel("log(1 + NT$)")
    fig.suptitle("Monthly payment amounts (log scale) — heavy zero mass", y=1.04, fontsize=13)
    save_fig(fig, "05_payment_amounts")


# --------------------------------------------------------------------------- #
# B. Bivariate — demographics
# --------------------------------------------------------------------------- #
def fig_categorical_default_rates(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    specs = [("SEX_LABEL", "Gender"), ("EDUCATION_LABEL", "Education"), ("MARRIAGE_LABEL", "Marital status")]
    for ax, (col, title) in zip(axes, specs):
        grp = df.groupby(col)[TARGET].agg(["mean", "size"]).reset_index()
        grp.columns = [col, "default_rate", "n"]
        sns.barplot(data=grp, x=col, y="default_rate", hue=col, legend=False,
                    palette="muted", ax=ax)
        for _, row in grp.iterrows():
            ax.text(row.name, row.default_rate + 0.004, f"{row.default_rate:.1%}\nn={int(row.n):,}",
                    ha="center", fontsize=9)
        ax.set_title(f"Default rate by {title.lower()}")
        ax.set_xlabel("")
        ax.set_ylabel("Default rate")
        ax.tick_params(axis="x", rotation=20)
    save_fig(fig, "06_categorical_default_rates")


def fig_repayment_status(df: pd.DataFrame) -> None:
    order = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    present = [v for v in order if v in df["PAY_0"].unique()]
    labels = [PAY_SCALE[v] for v in present]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    sns.countplot(data=df, x="PAY_0", hue=PAY_MONTHS["PAY_0"] and "PAY_0", legend=False,
                  order=present, palette="muted", ax=axes[0])
    axes[0].set_xticks(range(len(present)), labels, rotation=30, ha="right", fontsize=8)
    axes[0].set_title("Distribution of repayment status (Sep, PAY_0)")
    axes[0].set_xlabel("Repayment status")
    axes[0].set_ylabel("Clients")

    grp = df.groupby("PAY_0")[TARGET].mean().reindex(present)
    sns.barplot(x=grp.index.astype(str), y=grp.values, hue=grp.index.astype(str),
                legend=False, palette="Reds", ax=axes[1])
    for i, v in enumerate(grp.values):
        axes[1].text(i, v + 0.005, f"{v:.0%}", ha="center", fontsize=9)
    axes[1].set_xticks(range(len(present)), labels, rotation=30, ha="right", fontsize=8)
    axes[1].set_title("Default rate by repayment status (Sep, PAY_0)")
    axes[1].set_xlabel("Repayment status")
    axes[1].set_ylabel("Default rate")
    save_fig(fig, "07_repayment_status")


def fig_repayment_heatmap(df: pd.DataFrame) -> None:
    rows = []
    for col in PAY_STATUS_COLS:
        rate = df.groupby(col)[TARGET].mean()
        for status, r in rate.items():
            rows.append({"month": PAY_MONTHS[col], "status": status, "default_rate": r})
    heat = pd.DataFrame(rows).pivot(index="status", columns="month", values="default_rate")
    heat = heat.reindex(columns=["Apr", "May", "Jun", "Jul", "Aug", "Sep"])
    heat.index = [PAY_SCALE.get(s, str(s)) for s in heat.index]

    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    sns.heatmap(heat, annot=True, fmt=".1%", cmap="Reds", linewidths=0.5,
                vmin=0, vmax=0.85, cbar_kws={"label": "Default rate"}, ax=ax)
    ax.set_title("Default rate by repayment status across months")
    ax.set_xlabel("Month (Apr → Sep)")
    ax.set_ylabel("Repayment status")
    save_fig(fig, "08_repayment_heatmap")


def fig_boxplots_by_target(df: pd.DataFrame) -> None:
    specs = [("AGE", "Age (years)"), ("LIMIT_BAL", "Credit limit (NT$)"),
             ("BILL_AMT1", "Latest bill (NT$)"), ("PAY_AMT1", "Latest payment (NT$)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, (col, label) in zip(axes.ravel(), specs):
        sns.boxplot(data=df, x="TARGET_LABEL", y=col, hue="TARGET_LABEL", legend=False,
                    palette={"No default": COLOR_NO, "Default": COLOR_YES}, ax=ax,
                    fliersize=1.5, flierprops=dict(alpha=0.3))
        if col in ("LIMIT_BAL", "BILL_AMT1", "PAY_AMT1"):
            ax.set_yscale("symlog")
        ax.set_title(f"{col} by default outcome")
        ax.set_xlabel("")
        ax.set_ylabel(label)
    save_fig(fig, "09_boxplots_by_target")


def fig_violin_by_target(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, (col, label) in zip(axes, [("AGE", "Age (years)"), ("LIMIT_BAL", "Credit limit (NT$)")]):
        sns.violinplot(data=df, x="TARGET_LABEL", y=col, hue="TARGET_LABEL", legend=False,
                       palette={"No default": COLOR_NO, "Default": COLOR_YES},
                       inner="quartile", cut=0, ax=ax)
        ax.set_title(f"{label}: distribution shape by outcome")
        ax.set_xlabel("")
    save_fig(fig, "10_violin_by_target")


def fig_stacked_proportions(df: pd.DataFrame) -> None:
    specs = [("SEX_LABEL", "Gender"), ("EDUCATION_LABEL", "Education"), ("MARRIAGE_LABEL", "Marital status")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, (col, title) in zip(axes, specs):
        grp = df.groupby(col)[TARGET].agg(defaulted="mean", n="size").reset_index()
        grp["no_default"] = 1 - grp["defaulted"]
        ax.barh(grp[col], grp["no_default"], color=COLOR_NO, label="No default")
        ax.barh(grp[col], grp["defaulted"], left=grp["no_default"], color=COLOR_YES, label="Default")
        for y, (_, row) in enumerate(grp.iterrows()):
            ax.text(row.no_default / 2, y, f"{row.no_default:.0%}", ha="center", va="center",
                    color="white", fontsize=9)
            ax.text(row.no_default + row.defaulted / 2, y, f"{row.defaulted:.0%}", ha="center",
                    va="center", color="white", fontsize=9)
        ax.set_title(f"{title} — default share")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of clients")
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.7, -0.28), ncol=2)
    save_fig(fig, "11_stacked_proportions")


def fig_correlation_heatmap(df: pd.DataFrame) -> None:
    num_cols = ["LIMIT_BAL", "AGE", *PAY_STATUS_COLS, *BILL_COLS, *PAYAMT_COLS, TARGET]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 9.5), constrained_layout=True)
    sns.heatmap(corr, cmap="vlag", center=0, annot=False,
                cbar_kws={"label": "Pearson r"}, ax=ax,
                xticklabels=True, yticklabels=True)
    ax.set_title("Correlation matrix — raw numeric features + target")
    ax.tick_params(labelsize=8)
    save_fig(fig, "12_correlation_heatmap")


def _k_fmt(x: float) -> str:
    """Format an NT$ amount as a compact axis label (10k, 150k, 1M)."""
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M".replace(".0M", "M")
    val = x / 1_000
    return f"{val + 1e-9:.0f}k" if val >= 9.5 else f"{val:.1f}k"


def fig_limit_decile(df: pd.DataFrame) -> None:
    dec = df.assign(bin=pd.qcut(df["LIMIT_BAL"], 10, duplicates="drop"))
    grp = dec.groupby("bin", observed=True)[TARGET].agg(["mean", "size"]).reset_index()
    # Compact, readable bin labels: (9999.999, 30000.0] -> "10k–30k".
    grp["label"] = grp["bin"].map(
        lambda iv: f"{_k_fmt(iv.left)}–{_k_fmt(iv.right)}"
    )

    fig, ax = plt.subplots(figsize=(11, 4.4), constrained_layout=True)
    sns.barplot(data=grp, x="label", y="mean", hue="label", legend=False,
                palette=sns.color_palette("RdYlGn", len(grp)), ax=ax)
    for i, row in grp.iterrows():
        ax.text(i, row["mean"] + 0.003, f"{row['mean']:.0%}", ha="center", fontsize=9)
    ax.set_title("Default rate by credit-limit decile — higher limit, lower risk")
    ax.set_xlabel("LIMIT_BAL decile (NT$)")
    ax.set_ylabel("Default rate")
    ax.tick_params(axis="x", rotation=30)
    save_fig(fig, "13_limit_decile_default")


def eda_summary(df: pd.DataFrame) -> dict:
    """Key EDA facts reused by the dashboard / README."""
    by_pay0 = df.groupby("PAY_0")[TARGET].mean()
    return {
        "n_clients": int(len(df)),
        "default_rate": float(df[TARGET].mean()),
        "median_limit": float(df["LIMIT_BAL"].median()),
        "median_age": float(df["AGE"].median()),
        "default_rate_paid_duly": float(by_pay0.get(-1, np.nan)),
        "default_rate_2m_delay": float(by_pay0.get(2, np.nan)),
        "default_rate_by_sex": df.groupby("SEX")[TARGET].mean().round(4).to_dict(),
        "default_rate_by_education": df.groupby("EDUCATION")[TARGET].mean().round(4).to_dict(),
        "corr_limit_target": float(df["LIMIT_BAL"].corr(df[TARGET])),
        "corr_pay0_target": float(df["PAY_0"].corr(df[TARGET])),
    }


def run_all() -> None:
    set_style()
    raw = load_clean()
    df = _label_frame(raw)

    print("[eda] target / univariate ...")
    fig_target_distribution(df)
    fig_age_distribution(df)
    fig_limit_bal_distribution(df)
    fig_bill_amounts(df)
    fig_payment_amounts(df)

    print("[eda] bivariate ...")
    fig_categorical_default_rates(df)
    fig_repayment_status(df)
    fig_repayment_heatmap(df)
    fig_boxplots_by_target(df)
    fig_violin_by_target(df)
    fig_stacked_proportions(df)
    fig_correlation_heatmap(df)
    fig_limit_decile(df)

    summary = eda_summary(raw)
    (REPORTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[eda] summary -> {REPORTS_DIR / 'eda_summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_all()
