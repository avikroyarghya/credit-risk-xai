"""Shared utilities: project paths, plotting style, save helpers.

All modules in `src/` and the Streamlit app import from here so that
paths and visual identity stay consistent across the project.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for scripts / servers

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TUNING_DIR = REPORTS_DIR / "tuning"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, TUNING_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Plot identity
# --------------------------------------------------------------------------- #
# Consistent, colorblind-friendly palette used across EDA / evaluation / SHAP.
COLOR_NO = "#2A9D8F"   # teal  -> no default
COLOR_YES = "#E76F51"  # coral -> default
ACCENT = "#264653"
NEUTRAL = "#8D99AE"
TARGET_PALETTE = {0: COLOR_NO, 1: COLOR_YES}

RANDOM_STATE = 42


def set_style() -> None:
    """Apply the project-wide matplotlib/seaborn style."""
    sns.set_theme(style="whitegrid", context="notebook", palette="deep")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
            "legend.frameon": False,
        }
    )


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Save a matplotlib figure into reports/figures and return the path.

    Figures are expected to be created with constrained_layout=True; no
    tight_layout / bbox_inches tweaking is applied here on purpose.
    """
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
