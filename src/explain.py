"""Explainable AI, fairness audit and risk segmentation (Phases 14–16).

* SHAP TreeExplainer on the tuned XGBoost (global + local explanations).
  Driver *ranking* comes from the raw XGBoost; displayed probabilities come
  from the isotonic-calibrated model (calibration is a monotonic remapping,
  so it does not change the ranking of drivers).
* Fairness: observed performance differences across SEX / AGE / EDUCATION
  groups are reported with deliberately careful language — differences in
  this dataset do not by themselves establish discriminatory impact in a
  real lending institution.
* Risk segmentation: Low < 0.20 <= Medium < 0.50 <= High, with observed
  default rates per band and the share of actual defaulters captured.

Usage:
    python -m src.explain
"""
from __future__ import annotations

import json
import pickle

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.data_loader import EDUCATION_LABELS, SEX_LABELS, TARGET
from src.train import prepare_matrices
from src.utils import MODELS_DIR, REPORTS_DIR, save_fig, set_style

SHAP_SAMPLE = 3000
BANDS = [("Low", 0.0, 0.20), ("Medium", 0.20, 0.50), ("High", 0.50, 1.01)]


def _load_models() -> tuple:
    xgb_pipe = joblib.load(MODELS_DIR / "xgboost.joblib")
    calibrated = joblib.load(MODELS_DIR / "xgboost_calibrated.joblib")
    xgb_model = xgb_pipe.named_steps["clf"] if hasattr(xgb_pipe, "named_steps") else xgb_pipe
    return xgb_model, calibrated


# --------------------------------------------------------------------------- #
# SHAP — global explanations
# --------------------------------------------------------------------------- #
def global_explanations(xgb_model, X_sample: pd.DataFrame) -> shap.Explanation:
    explainer = shap.TreeExplainer(xgb_model)
    explanation = explainer(X_sample)  # Explanation with feature names
    return explainer, explanation


def fig_shap_global(explanation: shap.Explanation) -> pd.DataFrame:
    imp = pd.DataFrame(
        {"feature": explanation.feature_names,
         "mean_abs_shap": np.abs(explanation.values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(REPORTS_DIR / "shap_importance.csv", index=False)

    shap.plots.bar(explanation, max_display=15, show=False)
    save_fig(plt.gcf(), "20_shap_global_importance")

    shap.plots.beeswarm(explanation, max_display=15, show=False)
    save_fig(plt.gcf(), "21_shap_beeswarm")

    print("  top drivers:", imp.head(8).feature.tolist())
    return imp


def fig_shap_dependence(explanation: shap.Explanation, importance: pd.DataFrame) -> None:
    for i, feature in enumerate(importance.feature.head(3), start=22):
        shap.plots.scatter(explanation[:, feature], color=explanation, show=False)
        plt.title(f"SHAP dependence — {feature}")
        save_fig(plt.gcf(), f"{i}_shap_dependence_{feature}")


# --------------------------------------------------------------------------- #
# SHAP — local explanations
# --------------------------------------------------------------------------- #
def local_explanations(xgb_model, explainer, calibrated, data: dict) -> dict:
    X_te, y_te = data["X_test"], data["y_test"]
    proba = calibrated.predict_proba(X_te)[:, 1]

    # Prefer a clearly high-risk, non-saturated TRUE positive (isotonic caps at 1.0).
    hi_candidates = np.where((proba >= 0.70) & (proba <= 0.95) & (y_te.values == 1))[0]
    if len(hi_candidates):
        hi_idx = int(hi_candidates[np.argmax(proba[hi_candidates])])
    else:
        candidates = np.where((proba >= 0.70) & (proba <= 0.95))[0]
        hi_idx = int(candidates[np.argmax(proba[candidates])]) if len(candidates) else int(np.argmax(proba))
    hi_prob = float(proba[hi_idx])
    # Low-risk true negative, away from the isotonic floor of 0.
    lo_candidates = np.where((proba >= 0.02) & (proba <= 0.12) & (y_te.values == 0))[0]
    lo_idx = int(lo_candidates[np.argmin(proba[lo_candidates])]) if len(lo_candidates) else int(np.argmin(proba))
    lo_prob = float(proba[lo_idx])

    def waterfall_for(idx: int, prob: float, tag: str, fname: str) -> dict:
        row = X_te.iloc[[idx]]
        exp = explainer(row)
        shap.plots.waterfall(exp[0], max_display=12, show=False)
        plt.title(f"{tag} — calibrated default probability {prob:.0%}", fontsize=11)
        save_fig(plt.gcf(), fname)

        contrib = pd.DataFrame(
            {"feature": exp.feature_names, "value": row.values[0],
             "shap": exp.values[0]}
        ).reindex(np.abs(exp.values[0]).argsort()[::-1])
        return {
            "index": idx,
            "calibrated_probability": prob,
            "actual": int(y_te.iloc[idx]),
            "top_drivers": contrib.head(6).to_dict(orient="records"),
            "figure": fname,
        }

    print(f"  high-risk example: idx={hi_idx}, calibrated prob {hi_prob:.1%}")
    hi = waterfall_for(hi_idx, hi_prob, "High-risk customer", "25_shap_waterfall_high_risk")
    lo = waterfall_for(lo_idx, lo_prob, "Low-risk customer", "26_shap_waterfall_low_risk")
    return {"high_risk": hi, "low_risk": lo}


# --------------------------------------------------------------------------- #
# Fairness audit
# --------------------------------------------------------------------------- #
def fairness_audit(data: dict, proba: pd.Series, threshold: float) -> pd.DataFrame:
    X_te, y_te = data["X_test"], data["y_test"]
    pred = (proba >= threshold).astype(int)

    frames = []
    for col, name, mapper in [
        ("SEX", "Gender", SEX_LABELS),
        ("EDUCATION", "Education", EDUCATION_LABELS),
        ("AGE", "Age band", None),
    ]:
        if col == "AGE":
            group = pd.cut(X_te["AGE"], [20, 30, 45, 80], labels=["21–30", "31–45", "46+"],
                           include_lowest=True)
        else:
            group = X_te[col].map(mapper)
        for g in sorted(group.dropna().unique()):
            mask = (group == g).values
            tn, fp, fn, tp = confusion_matrix(y_te[mask], pred[mask], labels=[0, 1]).ravel()
            frames.append({
                "dimension": name, "group": str(g), "n": int(mask.sum()),
                "tpr_recall": round(tp / (tp + fn), 4) if (tp + fn) else np.nan,
                "fpr": round(fp / (fp + tn), 4) if (fp + tn) else np.nan,
                "precision": round(tp / (tp + fp), 4) if (tp + fp) else np.nan,
                "selection_rate": round(float(pred[mask].mean()), 4),
                "mean_pred_prob": round(float(proba[mask].mean()), 4),
                "observed_default_rate": round(float(y_te[mask].mean()), 4),
            })

    out = pd.DataFrame(frames)
    out.to_csv(REPORTS_DIR / "fairness_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for ax, metric in zip(axes, ["tpr_recall", "fpr", "selection_rate"]):
        sns.barplot(data=out, x="dimension", y=metric, hue="group", ax=ax, palette="muted")
        ax.set_title(metric.replace("_", " ").upper())
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(fontsize=8, title=None)
    fig.suptitle("Fairness audit — observed group-level performance differences (test set)", fontsize=12.5)
    save_fig(fig, "27_fairness_metrics")
    return out


# --------------------------------------------------------------------------- #
# Risk segmentation
# --------------------------------------------------------------------------- #
def risk_segmentation(y_te: pd.Series, proba: pd.Series) -> pd.DataFrame:
    band = pd.cut(proba, [b[1] for b in BANDS] + [1.01],
                  right=False, labels=[b[0] for b in BANDS])

    rows = []
    n_defaulters = int(y_te.sum())
    for name, lo, hi in BANDS:
        mask = (band == name).values
        n = int(mask.sum())
        defaults = int(y_te[mask].sum())
        rows.append({
            "band": name,
            "threshold": f"{lo:.0%} – {hi:.0%}" if name != "High" else f">= {lo:.0%}",
            "clients": n,
            "share": round(n / len(y_te), 4),
            "predicted_default_rate": round(float(proba[mask].mean()), 4),
            "observed_default_rate": round(defaults / n, 4) if n else np.nan,
            "defaulters_captured": round(defaults / n_defaulters, 4),
        })
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "risk_segments.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    colors = {"Low": "#2A9D8F", "Medium": "#E9C46A", "High": "#E76F51"}
    sns.barplot(data=out, x="band", y="clients", hue="band", legend=False,
                palette=colors, ax=axes[0])
    for i, row in out.iterrows():
        axes[0].text(i, row.clients + 60, f"{row.clients:,}\n({row.share:.1%})",
                     ha="center", fontsize=9)
    axes[0].set_title("Clients per risk band")
    axes[0].set_ylabel("Clients (test set)")

    sns.barplot(data=out, x="band", y="observed_default_rate", hue="band", legend=False,
                palette=colors, ax=axes[1])
    for i, row in out.iterrows():
        axes[1].text(i, row.observed_default_rate + 0.01,
                     f"obs {row.observed_default_rate:.1%}\npred {row.predicted_default_rate:.1%}",
                     ha="center", fontsize=9)
    axes[1].set_title("Observed vs predicted default rate per band")
    axes[1].set_ylabel("Default rate")
    axes[1].set_ylim(0, max(out.observed_default_rate) * 1.3)
    save_fig(fig, "28_risk_segments")
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    set_style()
    print("[explain] preparing matrices ...")
    data = prepare_matrices()
    X_te, y_te = data["X_test"], data["y_test"]
    xgb_model, calibrated = _load_models()

    print("[explain] SHAP global explanations ...")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_te), size=min(SHAP_SAMPLE, len(X_te)), replace=False)
    X_sample = X_te.iloc[sample_idx]
    explainer, explanation = global_explanations(xgb_model, X_sample)
    importance = fig_shap_global(explanation)

    print("[explain] SHAP dependence plots ...")
    fig_shap_dependence(explanation, importance)

    print("[explain] SHAP local explanations ...")
    local = local_explanations(xgb_model, explainer, calibrated, data)

    print("[explain] fairness audit ...")
    proba_te = pd.Series(calibrated.predict_proba(X_te)[:, 1], index=X_te.index)
    eval_summary = json.loads((REPORTS_DIR / "eval_summary.json").read_text())
    fairness = fairness_audit(data, proba_te, eval_summary["threshold"])
    print(fairness.to_string(index=False))

    print("[explain] risk segmentation ...")
    segments = risk_segmentation(y_te, proba_te)
    print(segments.to_string(index=False))

    bundle = {
        "X_sample": X_sample,
        "shap_values": explanation.values,
        "expected_value": float(np.asarray(explainer.expected_value).ravel()[0]),
        "calibrated_proba": pd.Series(
            calibrated.predict_proba(X_sample)[:, 1], index=X_sample.index),
        "y_sample": y_te.iloc[sample_idx],
        "importance": importance,
        "local": local,
        "segments": segments,
        "fairness": fairness,
    }
    with open(REPORTS_DIR / "shap_bundle.pkl", "wb") as fh:
        pickle.dump(bundle, fh)

    highlights = {
        "top_5_features": importance.feature.head(5).tolist(),
        "high_risk_probability": local["high_risk"]["calibrated_probability"],
        "high_risk_top_drivers": [d["feature"] for d in local["high_risk"]["top_drivers"][:4]],
        "band_observed_rates": segments.set_index("band").observed_default_rate.to_dict(),
        "high_band_capture": float(segments.loc[segments.band == "High", "defaulters_captured"].iloc[0]),
    }
    (REPORTS_DIR / "xai_highlights.json").write_text(json.dumps(highlights, indent=2))
    print(f"[explain] done -> {REPORTS_DIR / 'xai_highlights.json'}")


if __name__ == "__main__":
    main()
