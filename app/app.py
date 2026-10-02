"""Streamlit dashboard — Credit Risk Prediction & Explainable AI.

Pages
-----
1. Overview            — KPIs, selected model, risk-band distribution
2. EDA                 — key exploratory figures + data preview
3. Model Comparison    — metric table + ROC / PR / calibration curves
4. Risk Predictor      — customer inputs -> calibrated default probability,
                          risk band + live SHAP waterfall
5. Explainability      — global SHAP, dependence, fairness, segmentation

Run from the project root:
    streamlit run app/app.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (  # noqa: E402
    EDUCATION_LABELS,
    MARRIAGE_LABELS,
    PAY_MONTHS,
    SEX_LABELS,
    load_clean,
)
from src.feature_engineering import add_features  # noqa: E402
from src.preprocessing import load_splits  # noqa: E402
from src.utils import FIGURES_DIR, MODELS_DIR, REPORTS_DIR  # noqa: E402

st.set_page_config(
    page_title="Credit Risk Prediction & Explainable AI",
    page_icon="🏦",
    layout="wide",
)

BAND_EDGES = [0.0, 0.20, 0.50, 1.01]
BAND_NAMES = ["Low", "Medium", "High"]
BAND_COLORS = {"Low": "#2A9D8F", "Medium": "#E9C46A", "High": "#E76F51"}
PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_artifacts() -> dict:
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text())
    eval_summary = json.loads((REPORTS_DIR / "eval_summary.json").read_text())
    eda_summary = json.loads((REPORTS_DIR / "eda_summary.json").read_text())
    xai = json.loads((REPORTS_DIR / "xai_highlights.json").read_text())

    calibrated = joblib.load(MODELS_DIR / "xgboost_calibrated.joblib")
    xgb_pipe = joblib.load(MODELS_DIR / "xgboost.joblib")
    xgb_model = xgb_pipe.named_steps["clf"] if hasattr(xgb_pipe, "named_steps") else xgb_pipe
    explainer = shap.TreeExplainer(xgb_model)

    results = pd.read_csv(REPORTS_DIR / "model_results.csv")
    segments = pd.read_csv(REPORTS_DIR / "risk_segments.csv")
    fairness = pd.read_csv(REPORTS_DIR / "fairness_metrics.csv")
    importance = pd.read_csv(REPORTS_DIR / "shap_importance.csv")
    with open(REPORTS_DIR / "shap_bundle.pkl", "rb") as fh:
        bundle = pickle.load(fh)

    return {
        "metadata": metadata, "eval": eval_summary, "eda": eda_summary, "xai": xai,
        "calibrated": calibrated, "xgb": xgb_model, "explainer": explainer,
        "results": results, "segments": segments, "fairness": fairness,
        "importance": importance, "bundle": bundle,
        "feature_cols": metadata["feature_cols"],
    }


@st.cache_data
def load_data() -> pd.DataFrame:
    return load_clean()


@st.cache_data
def load_test() -> pd.DataFrame:
    return load_splits()["test"]


@st.cache_data
def fig_bytes(name: str):
    from PIL import Image

    path = FIGURES_DIR / f"{name}.png"
    return Image.open(path) if path.exists() else None


def assign_band(prob: float) -> str:
    for name, lo, hi in zip(BAND_NAMES, BAND_EDGES[:-1], BAND_EDGES[1:]):
        if lo <= prob < hi:
            return name
    return "High"


def risk_meter(prob: float) -> None:
    """Horizontal risk thermometer with band zones."""
    fig, ax = plt.subplots(figsize=(7, 1.6), constrained_layout=True)
    for name, lo, hi in zip(BAND_NAMES, BAND_EDGES[:-1], BAND_EDGES[1:]):
        ax.barh([0], [hi - lo], left=lo, height=0.5, color=BAND_COLORS[name], alpha=0.85)
    ax.axvline(prob, color="#264653", lw=3)
    ax.text(prob, 0.42, f"{prob:.1%}", ha="center", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.2, 0.5, 1.0], ["0%", "20%", "50%", "100%"])
    ax.set_title(f"Predicted default probability — risk band: {assign_band(prob)}",
                 fontsize=11, fontweight="bold")
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Page 1 — Overview
# --------------------------------------------------------------------------- #
def page_overview(a: dict) -> None:
    st.header("📊 Project Overview")
    st.caption(
        "Predicting Credit Default Risk Using Statistical Learning, "
        "Gradient Boosting and SHAP — UCI Default of Credit Card Clients."
    )

    e, ev, md = a["eda"], a["eval"], a["metadata"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Clients analysed", f"{e['n_clients']:,}")
    k2.metric("Default rate", f"{e['default_rate']:.1%}")
    k3.metric("Selected model", "XGBoost (isotonic)")
    k4.metric("Test ROC-AUC", f"{ev['test_roc_auc']:.3f}")
    k5.metric("Test PR-AUC", f"{ev['test_pr_auc']:.3f}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Key facts")
        st.markdown(
            f"""
            - **23 raw features + 19 engineered features** feed the final model
              (42 model inputs in total).
            - **Imbalance handling:** {md['imbalance']['selected_strategy_xgb']}
              — selected on validation PR-AUC (class ratio ≈
              {1 / md['imbalance']['scale_pos_weight']:.2f} : 1).
            - **Decision threshold:** {ev['threshold']:.2f} (max validation F1)
              → test recall **{ev['test_recall']:.1%}**,
              precision **{ev['test_precision']:.1%}**.
            - **Calibration:** isotonic; test Brier **{ev['test_brier']:.4f}**.
            - Top SHAP drivers: {', '.join(a['xai']['top_5_features'][:5])}.
            """
        )
        st.subheader("Risk-band distribution (test set)")
        seg = a["segments"]
        fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
        colors = [BAND_COLORS[b] for b in seg.band]
        ax.bar(seg.band, seg.share * 100, color=colors)
        for i, row in seg.iterrows():
            ax.text(i, row.share * 100 + 1.2,
                    f"{row.clients:,} clients\nobserved default {row.observed_default_rate:.1%}",
                    ha="center", fontsize=9)
        ax.set_ylabel("% of test clients")
        ax.set_ylim(0, 82)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Project architecture")
        st.code(
            "UCI dataset → cleaning → EDA → statistical tests\n"
            "  → feature engineering → stratified split\n"
            "      ├─ Logistic Regression (baseline)\n"
            "      └─ XGBoost (tuned + calibrated)\n"
            "  → imbalance handling → evaluation\n"
            "  → SHAP / XAI → fairness audit\n"
            "  → risk segmentation → this dashboard",
            language="text",
        )
        st.info(
            "From a risk-management perspective, the *estimated probability of "
            "default* is more valuable than a bare binary credible/not-credible "
            "classification (Yeh & Lien, 2009).",
            icon="💡",
        )


# --------------------------------------------------------------------------- #
# Page 2 — EDA
# --------------------------------------------------------------------------- #
def page_eda(a: dict) -> None:
    st.header("🔍 Exploratory Data Analysis")
    df = load_data()

    figs = {
        "Target distribution": "01_target_distribution",
        "Age vs default": "02_age_distribution",
        "Credit limit vs default": "03_limit_bal_distribution",
        "Monthly bill amounts": "04_bill_amounts",
        "Monthly payments (log)": "05_payment_amounts",
        "Default rate by demographics": "06_categorical_default_rates",
        "Repayment status vs default": "07_repayment_status",
        "Default rate by month × status": "08_repayment_heatmap",
        "Boxplots by outcome": "09_boxplots_by_target",
        "Stacked proportions": "11_stacked_proportions",
        "Correlation heatmap": "12_correlation_heatmap",
        "Default rate by limit decile": "13_limit_decile_default",
    }
    choice = st.selectbox("Select a figure", list(figs.keys()))
    img = fig_bytes(figs[choice])
    if img is not None:
        st.image(img, use_container_width=True)

    st.subheader("Data preview")
    st.dataframe(df.head(200), use_container_width=True)

    with st.expander("Summary statistics"):
        st.dataframe(df.describe().T.round(1), use_container_width=True)


# --------------------------------------------------------------------------- #
# Page 3 — Model comparison
# --------------------------------------------------------------------------- #
def page_models(a: dict) -> None:
    st.header("⚖️ Model Comparison")
    st.caption("All metrics on the held-out test set (15% of clients, never used for tuning).")

    st.subheader("Leaderboard (sorted by PR-AUC)")
    show = a["results"].copy()
    st.dataframe(
        show.style.format(
            {c: "{:.3f}" for c in ["accuracy", "precision", "recall", "f1",
                                   "roc_auc", "pr_auc", "brier"]}
        ).background_gradient(subset=["roc_auc", "pr_auc", "f1"], cmap="Greens"),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.image(fig_bytes("16_roc_pr_curves"), use_container_width=True)
        st.image(fig_bytes("19_threshold_analysis"), use_container_width=True)
    with c2:
        st.image(fig_bytes("17_confusion_matrices"), use_container_width=True)
        st.image(fig_bytes("18_calibration_curves"), use_container_width=True)

    st.markdown(
        f"""**Reading the confusion matrix in business terms** — a *false negative*
        is a customer who actually defaults but was scored low-risk (missed loss);
        a *false positive* is a good customer denied or over-priced. At the tuned
        threshold the model catches **{a['eval']['test_recall']:.0%}** of actual
        defaulters in the test set."""
    )

    with st.expander("Imbalance-handling experiments (validation)"):
        imb = pd.DataFrame(a["metadata"]["imbalance"]["experiments"])
        st.dataframe(imb, use_container_width=True)

    with st.expander("XGBoost best hyperparameters"):
        st.json(a["metadata"]["xgb_best_params"])


# --------------------------------------------------------------------------- #
# Page 4 — Customer risk predictor
# --------------------------------------------------------------------------- #
def page_predictor(a: dict) -> None:
    st.header("🎯 Customer Risk Predictor")
    st.caption(
        "Enter a customer profile → calibrated default probability, risk band and "
        "the SHAP drivers behind the prediction. Repayment status codes: "
        "−2 = no consumption, −1 = paid duly, 0 = revolving, 1–9 = months of delay."
    )

    df, test = load_data(), load_test()
    b = a["bundle"]

    examples = {
        "— Custom input —": None,
        "Median customer": "median",
        "Random customer": "random",
        "High-risk showcase (93% risk)": "high",
        "Low-risk showcase (2% risk)": "low",
    }
    choice = st.selectbox("Load an example profile", list(examples.keys()))
    if st.button("Apply example", type="primary"):
        st.session_state.example_tick = st.session_state.get("example_tick", 0) + 1
        st.session_state.example_choice = choice
    tick = st.session_state.get("example_tick", 0)
    chosen = st.session_state.get("example_choice", "— Custom input —")

    if chosen == "Random customer":
        base = df.sample(1, random_state=7 + tick).iloc[0]
    elif chosen == "High-risk showcase (93% risk)":
        base = test.iloc[b["local"]["high_risk"]["index"]]
    elif chosen == "Low-risk showcase (2% risk)":
        base = test.iloc[b["local"]["low_risk"]["index"]]
    else:  # custom input or median customer -> start from the median profile
        base = df.median(numeric_only=True)

    def clipped(col: str, lo: int, hi: int) -> int:
        """Widget default, safely clipped into the allowed bounds."""
        return int(np.clip(float(base[col]), lo, hi))

    col_in, col_out = st.columns([3, 2], gap="large")

    raw: dict = {}
    with col_in:
        st.markdown("#### Customer profile")
        c1, c2, c3 = st.columns(3)
        raw["LIMIT_BAL"] = c1.number_input(
            "Credit limit (NT$)", 10_000, 1_000_000,
            clipped("LIMIT_BAL", 10_000, 1_000_000), 10_000, key=f"limit_{tick}")
        raw["AGE"] = c2.number_input(
            "Age", 21, 79, clipped("AGE", 21, 79), 1, key=f"age_{tick}")
        sex_opts = list(SEX_LABELS.values())
        raw["SEX"] = int(list(SEX_LABELS)[
            sex_opts.index(c3.selectbox("Gender", sex_opts,
                                        index=int(np.clip(base["SEX"], 1, 2)) - 1,
                                        key=f"sex_{tick}"))])
        edu_opts = list(EDUCATION_LABELS.values())
        raw["EDUCATION"] = int(list(EDUCATION_LABELS)[
            edu_opts.index(c1.selectbox("Education", edu_opts,
                                        index=int(np.clip(base["EDUCATION"], 1, 4)) - 1,
                                        key=f"edu_{tick}"))])
        mar_opts = list(MARRIAGE_LABELS.values())
        raw["MARRIAGE"] = int(list(MARRIAGE_LABELS)[
            mar_opts.index(c2.selectbox("Marital status", mar_opts,
                                        index=int(np.clip(base["MARRIAGE"], 1, 3)) - 1,
                                        key=f"mar_{tick}"))])

        st.markdown("#### Repayment status (last 6 months)")
        cols = st.columns(3)
        for i, pay_col in enumerate(PAY_STATUS_COLS):
            raw[pay_col] = cols[i % 3].number_input(
                f"{PAY_MONTHS[pay_col]}", -2, 9, clipped(pay_col, -2, 9), 1,
                key=f"{pay_col}_{tick}")
        st.markdown("#### Bill statement amounts (NT$)")
        cols = st.columns(3)
        for i, bill_col in enumerate(BILL_COLS):
            raw[bill_col] = cols[i % 3].number_input(
                f"{PAY_MONTHS[PAY_STATUS_COLS[i]]} bill", -100_000, 1_000_000,
                clipped(bill_col, -100_000, 1_000_000), 1_000, key=f"{bill_col}_{tick}")
        st.markdown("#### Amounts paid (NT$)")
        cols = st.columns(3)
        for i, pay_amt in enumerate(PAYAMT_COLS):
            raw[pay_amt] = cols[i % 3].number_input(
                f"{PAY_MONTHS[PAY_STATUS_COLS[i]]} payment", 0, 1_000_000,
                clipped(pay_amt, 0, 1_000_000), 1_000, key=f"{pay_amt}_{tick}")

    with col_out:
        st.markdown("#### Prediction")
        row = pd.DataFrame([raw])
        row_fe = add_features(row)
        X = row_fe[a["feature_cols"]].astype(float)
        prob = float(a["calibrated"].predict_proba(X)[0, 1])

        band = assign_band(prob)
        band_color = {"Low": "green", "Medium": "orange", "High": "red"}[band]
        st.markdown(
            f"<span style='font-size:1.6rem; font-weight:800'>"
            f"Default probability: {prob:.1%}</span><br>"
            f"Risk band: <span style='color:{band_color}; font-weight:700'>{band}</span>",
            unsafe_allow_html=True,
        )
        risk_meter(prob)

        exp = a["explainer"](X)
        contrib = pd.DataFrame(
            {"feature": exp.feature_names, "value": X.values[0],
             "impact": exp.values[0]}
        ).reindex(np.abs(exp.values[0]).argsort()[::-1]).head(8)
        contrib["direction"] = np.where(contrib.impact > 0, "↑ risk", "↓ risk")
        st.markdown("##### Top contributing factors (SHAP)")
        st.dataframe(
            contrib[["feature", "value", "impact", "direction"]]
            .style.format({"value": "{:,.0f}", "impact": "{:+.2f}"}),
            use_container_width=True, height=320,
        )

        st.markdown("##### SHAP waterfall (this customer)")
        shap.plots.waterfall(exp[0], max_display=10, show=False)
        st.pyplot(plt.gcf())
        plt.close("all")


# --------------------------------------------------------------------------- #
# Page 5 — Explainability
# --------------------------------------------------------------------------- #
def page_explain(a: dict) -> None:
    st.header("🔬 Explainable AI (SHAP)")
    st.caption(
        "Driver ranking from the tuned XGBoost; probabilities from the "
        "isotonic-calibrated model."
    )

    st.subheader("Global feature importance (mean |SHAP|)")
    c1, c2 = st.columns(2)
    with c1:
        st.image(fig_bytes("20_shap_global_importance"), use_container_width=True)
    with c2:
        st.image(fig_bytes("21_shap_beeswarm"), use_container_width=True)

    st.subheader("Dependence plots — top 3 features")
    for i, feature in enumerate(a["importance"].feature.head(3)):
        name = [f"{n}_shap_dependence_{feature}" for n in (22, 23, 24)][i]
        st.image(fig_bytes(name), use_container_width=True)

    st.subheader("Local explanations")
    c1, c2 = st.columns(2)
    with c1:
        st.image(fig_bytes("25_shap_waterfall_high_risk"), use_container_width=True)
    with c2:
        st.image(fig_bytes("26_shap_waterfall_low_risk"), use_container_width=True)

    st.subheader("Fairness audit — observed group differences")
    st.caption(
        "Observed performance differences in this dataset do not by themselves "
        "establish discriminatory impact in a real lending institution; small "
        "groups (e.g. Education = Others, n=64) give unstable estimates."
    )
    st.dataframe(a["fairness"], use_container_width=True)

    st.subheader("Risk segmentation (test set)")
    st.dataframe(a["segments"], use_container_width=True)
    st.image(fig_bytes("28_risk_segments"), use_container_width=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.sidebar.title("🏦 Credit Risk XAI")
    st.sidebar.caption("UCI Default of Credit Card Clients — 29,965 clients")
    page = st.sidebar.radio(
        "Navigate",
        ["📊 Overview", "🔍 EDA", "⚖️ Model Comparison",
         "🎯 Risk Predictor", "🔬 Explainability"],
    )
    a = load_artifacts()

    if page.startswith("📊"):
        page_overview(a)
    elif page.startswith("🔍"):
        page_eda(a)
    elif page.startswith("⚖"):
        page_models(a)
    elif page.startswith("🎯"):
        page_predictor(a)
    else:
        page_explain(a)

    st.sidebar.divider()
    st.sidebar.markdown(
        f"**Test performance**  \nROC-AUC {a['eval']['test_roc_auc']:.3f} · "
        f"PR-AUC {a['eval']['test_pr_auc']:.3f}  \nBrier {a['eval']['test_brier']:.4f}"
    )


if __name__ == "__main__":
    main()
