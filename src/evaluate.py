"""Model evaluation on the held-out test set (Phase 12–13).

* Threshold is tuned on VALIDATION predictions of a train-fitted clone of the
  final calibrated model (no test leakage), then applied to the test set.
* All models are compared on: accuracy, precision, recall, F1, ROC-AUC,
  PR-AUC (average precision) and Brier score.
* Probability quality is shown with reliability (calibration) curves.
* Confusion matrices are explained in business terms (FN = missed defaulter).

Usage:
    python -m src.evaluate
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

from src.data_loader import TARGET
from src.train import prepare_matrices, xgb_classifier
from src.utils import MODELS_DIR, REPORTS_DIR, save_fig, set_style

MODEL_FILES = {
    "dummy": "Dummy (prior)",
    "logistic": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost (tuned)",
    "xgboost_calibrated": "XGBoost (calibrated)",
}
MAIN_MODEL = "xgboost_calibrated"


# --------------------------------------------------------------------------- #
# Threshold selection (validation, train-fitted clone — honest)
# --------------------------------------------------------------------------- #
def select_threshold(metadata: dict, data: dict) -> tuple[float, pd.DataFrame]:
    """Refit the calibrated model on TRAIN only, sweep thresholds on VAL."""
    p = metadata["xgb_best_params"]
    strategy = metadata["imbalance"]["selected_strategy_xgb"]
    spw = metadata["imbalance"]["scale_pos_weight"]

    base = xgb_classifier(strategy, spw=spw, n_jobs=2, **p)
    if strategy == "smote":
        base = ImbPipeline([("smote", SMOTE(random_state=42)), ("clf", base)])
    cal = CalibratedClassifierCV(base, method=metadata["calibration"]["method"], cv=5)
    cal.fit(data["X_train"], data["y_train"])

    proba = cal.predict_proba(data["X_val"])[:, 1]
    y_val = data["y_val"]

    prec, rec, f1s, ths = [], [], [], []
    for th in np.linspace(0.05, 0.95, 181):
        pred = (proba >= th).astype(int)
        prec.append(precision_score(y_val, pred, zero_division=0))
        rec.append(recall_score(y_val, pred, zero_division=0))
        f1s.append(f1_score(y_val, pred, zero_division=0))
        ths.append(th)

    sweep = pd.DataFrame({"threshold": ths, "precision": prec, "recall": rec, "f1": f1s})
    best = float(sweep.loc[sweep.f1.idxmax(), "threshold"])
    print(f"  tuned threshold (max validation F1): {best:.2f}")
    return best, sweep


# --------------------------------------------------------------------------- #
# Test-set evaluation
# --------------------------------------------------------------------------- #
def evaluate_all(data: dict, threshold: float) -> tuple[pd.DataFrame, dict]:
    X_te, y_te = data["X_test"], data["y_test"]
    rows, probas = [], {}

    for key, label in MODEL_FILES.items():
        model = joblib.load(MODELS_DIR / f"{key}.joblib")
        proba = model.predict_proba(X_te)[:, 1]
        probas[key] = proba

        th = threshold if key == MAIN_MODEL else 0.5
        pred = (proba >= th).astype(int)
        rows.append(
            {
                "model": label,
                "threshold": th,
                "accuracy": round(accuracy_score(y_te, pred), 4),
                "precision": round(precision_score(y_te, pred, zero_division=0), 4),
                "recall": round(recall_score(y_te, pred, zero_division=0), 4),
                "f1": round(f1_score(y_te, pred, zero_division=0), 4),
                "roc_auc": round(roc_auc_score(y_te, proba), 4),
                "pr_auc": round(average_precision_score(y_te, proba), 4),
                "brier": round(brier_score_loss(y_te, proba), 4),
            }
        )

    table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    table.to_csv(REPORTS_DIR / "model_results.csv", index=False)
    print(table.to_string(index=False))

    main_pred = (probas[MAIN_MODEL] >= threshold).astype(int)
    report = classification_report(y_te, main_pred, target_names=["No default", "Default"])
    (REPORTS_DIR / "classification_report.txt").write_text(report)

    summary = {
        "threshold": threshold,
        "test_roc_auc": float(table.loc[table.model == MODEL_FILES[MAIN_MODEL], "roc_auc"].iloc[0]),
        "test_pr_auc": float(table.loc[table.model == MODEL_FILES[MAIN_MODEL], "pr_auc"].iloc[0]),
        "test_brier": float(table.loc[table.model == MODEL_FILES[MAIN_MODEL], "brier"].iloc[0]),
        "test_recall": float(table.loc[table.model == MODEL_FILES[MAIN_MODEL], "recall"].iloc[0]),
        "test_precision": float(table.loc[table.model == MODEL_FILES[MAIN_MODEL], "precision"].iloc[0]),
        "confusion": confusion_matrix(y_te, main_pred).tolist(),
    }
    return table, {"probas": probas, "summary": summary, "report": report}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_curves(data: dict, probas: dict) -> None:
    y_te = data["y_test"]
    colors = sns.color_palette("deep", len(MODEL_FILES))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
    for (key, label), color in zip(MODEL_FILES.items(), colors):
        fpr, tpr, _ = roc_curve(y_te, probas[key])
        axes[0].plot(fpr, tpr, color=color, lw=1.8,
                     label=f"{label} (AUC {roc_auc_score(y_te, probas[key]):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curves — test set")
    axes[0].legend(fontsize=8.5)

    for (key, label), color in zip(MODEL_FILES.items(), colors):
        prec, rec, _ = precision_recall_curve(y_te, probas[key])
        axes[1].plot(rec, prec, color=color, lw=1.8,
                     label=f"{label} (AP {average_precision_score(y_te, probas[key]):.3f})")
    axes[1].axhline(y_te.mean(), color="k", ls="--", lw=1, alpha=0.5,
                    label=f"Prevalence ({y_te.mean():.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall curves — test set")
    axes[1].legend(fontsize=8.5, loc="upper right")
    save_fig(fig, "16_roc_pr_curves")


def fig_confusion(data: dict, probas: dict, threshold: float) -> None:
    y_te = data["y_test"]
    specs = [
        ("xgboost_calibrated", threshold, f"XGBoost calibrated @ tuned threshold {threshold:.2f}"),
        ("logistic", 0.5, "Logistic Regression @ 0.50"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for ax, (key, th, title) in zip(axes, specs):
        pred = (probas[key] >= th).astype(int)
        cm = confusion_matrix(y_te, pred)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=cm, fmt=",d", cmap="Blues", cbar=False,
                    vmin=0, vmax=1, annot_kws={"fontsize": 11}, ax=ax)
        ax.set_xticks([0.5, 1.5], ["Pred: No default", "Pred: Default"])
        ax.set_yticks([0.5, 1.5], ["Actual: No", "Actual: Default"], rotation=0)
        ax.set_title(title, fontsize=11)
    fig.suptitle("Confusion matrices — test set (counts + row-normalized)", fontsize=13)
    save_fig(fig, "17_confusion_matrices")


def fig_calibration(data: dict, probas: dict) -> None:
    y_te = data["y_test"]
    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    for key, label in [("dummy", "Dummy (prior)"), ("logistic", "Logistic Regression"),
                       ("xgboost", "XGBoost (tuned)"), ("xgboost_calibrated", "XGBoost (calibrated)")]:
        frac_pos, mean_pred = calibration_curve(y_te, probas[key], n_bins=10,
                                                strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", ms=4, lw=1.8,
                label=f"{label} (Brier {brier_score_loss(y_te, probas[key]):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of actual defaults")
    ax.set_title("Reliability (calibration) curves — test set")
    ax.legend(fontsize=9)
    save_fig(fig, "18_calibration_curves")


def fig_threshold_sweep(sweep: pd.DataFrame, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.plot(sweep.threshold, sweep.precision, label="Precision", color="#2A9D8F")
    ax.plot(sweep.threshold, sweep.recall, label="Recall", color="#E76F51")
    ax.plot(sweep.threshold, sweep.f1, label="F1", color="#264653", lw=2.2)
    ax.axvline(threshold, color="#8D99AE", ls="--", lw=1.5)
    ax.text(threshold + 0.01, 0.45, f"chosen\n{threshold:.2f}", fontsize=9)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score (validation)")
    ax.set_title("Threshold tuning — precision / recall / F1 vs threshold")
    ax.legend()
    save_fig(fig, "19_threshold_analysis")


# --------------------------------------------------------------------------- #
def main() -> None:
    set_style()
    print("[evaluate] preparing matrices ...")
    data = prepare_matrices()
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text())

    print("[evaluate] tuning decision threshold on validation ...")
    threshold, sweep = select_threshold(metadata, data)
    sweep.to_csv(REPORTS_DIR / "threshold_sweep.csv", index=False)

    print("[evaluate] test-set metrics for all models ...")
    table, bundle = evaluate_all(data, threshold)

    print("[evaluate] figures ...")
    fig_curves(data, bundle["probas"])
    fig_confusion(data, bundle["probas"], threshold)
    fig_calibration(data, bundle["probas"])
    fig_threshold_sweep(sweep, threshold)

    (REPORTS_DIR / "eval_summary.json").write_text(json.dumps(bundle["summary"], indent=2))
    print("\n" + bundle["report"])
    print(f"[evaluate] summary -> {REPORTS_DIR / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
