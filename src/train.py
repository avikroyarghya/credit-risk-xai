"""Model training (Phases 8–11).

Pipeline stages
---------------
1. Baselines    : DummyClassifier, LogisticRegression (interpretable)
2. Advanced ML  : DecisionTree, RandomForest, XGBoost
3. Imbalance    : none vs class-weights vs SMOTE, selected on validation PR-AUC
                  (resampling is fitted inside the training data only — no leakage)
4. Tuning       : RandomizedSearchCV on XGBoost (scoring = average precision)
5. Calibration  : sigmoid (Platt) vs isotonic, selected on validation Brier score
6. Final models : refit on train+val, persisted to models/ for the test-set
                  evaluation in src/evaluate.py — the test set is never used here.

Usage:
    python -m src.train
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data_loader import TARGET
from src.feature_engineering import CATEGORICAL_RAW, add_features, feature_columns
from src.preprocessing import load_splits
from src.utils import MODELS_DIR, RANDOM_STATE, TUNING_DIR

warnings.filterwarnings("ignore", category=UserWarning)

SEED = RANDOM_STATE
N_SEARCH_ITER = 15
SEARCH_CV = 3


# --------------------------------------------------------------------------- #
# Data / feature matrices
# --------------------------------------------------------------------------- #
def prepare_matrices() -> dict:
    """Load splits, engineer features, return X/y per split + column list."""
    splits = load_splits()
    feats = {name: add_features(df) for name, df in splits.items()}

    cols = feature_columns(feats["train"])
    numeric = [c for c in cols if c not in CATEGORICAL_RAW]

    out = {"feature_cols": cols, "numeric_cols": numeric, "cat_cols": CATEGORICAL_RAW}
    for name, df in feats.items():
        out[f"X_{name}"] = df[cols].astype(float)
        out[f"y_{name}"] = df[TARGET].astype(int)
    return out


def lr_pipeline(strategy: str, numeric: list[str]) -> ImbPipeline:
    """Logistic regression with one-hot + scaling; strategy in {none, weight, smote}."""
    preprocess = ColumnTransformer(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL_RAW),
            ("scale", StandardScaler(), numeric),
        ],
        remainder="drop",
    )
    cw = "balanced" if strategy == "weight" else None
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight=cw, random_state=SEED)
    steps = [("prep", preprocess)]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("clf", clf))
    return ImbPipeline(steps)


def xgb_classifier(strategy: str, spw: float | None = None, **params) -> XGBClassifier:
    """XGBoost with an imbalance-handling strategy."""
    base = dict(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        tree_method="hist",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=2,
    )
    base.update(params)
    if strategy == "weight":
        base["scale_pos_weight"] = spw
    return XGBClassifier(**base)


def xgb_pipeline(strategy: str, spw: float) -> ImbPipeline | XGBClassifier:
    """XGBoost wrapped for optional SMOTE resampling (train-side only)."""
    clf = xgb_classifier(strategy, spw=spw)
    if strategy == "smote":
        return ImbPipeline([("smote", SMOTE(random_state=SEED)), ("clf", clf)])
    return clf


# --------------------------------------------------------------------------- #
# Validation metrics
# --------------------------------------------------------------------------- #
def val_metrics(model, X: pd.DataFrame, y: pd.Series) -> dict:
    proba = model.predict_proba(X)[:, 1]
    return {
        "val_roc_auc": round(roc_auc_score(y, proba), 4),
        "val_pr_auc": round(average_precision_score(y, proba), 4),
        "val_brier": round(brier_score_loss(y, proba), 4),
    }


# --------------------------------------------------------------------------- #
# Stage 3 — imbalance experiments
# --------------------------------------------------------------------------- #
def run_imbalance_experiments(data: dict) -> tuple[pd.DataFrame, dict]:
    """Compare none / class-weight / SMOTE for LR and XGB on the validation set."""
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_va, y_va = data["X_val"], data["y_val"]
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())

    rows = []
    for family, builder in {
        "logistic": lambda s: lr_pipeline(s, data["numeric_cols"]),
        "xgboost": lambda s: xgb_pipeline(s, spw),
    }.items():
        for strategy in ["none", "weight", "smote"]:
            t0 = time.time()
            model = builder(strategy)
            model.fit(X_tr, y_tr)
            m = val_metrics(model, X_va, y_va)
            rows.append({"model": family, "strategy": strategy, **m,
                         "fit_seconds": round(time.time() - t0, 1)})
            print(f"  {family:9s} | {strategy:7s} | PR-AUC {m['val_pr_auc']:.4f} "
                  f"| ROC-AUC {m['val_roc_auc']:.4f} | Brier {m['val_brier']:.4f}")

    table = pd.DataFrame(rows)
    table.to_csv(TUNING_DIR / "imbalance_experiments.csv", index=False)

    chosen = {}
    for family in ["logistic", "xgboost"]:
        sub = table[table.model == family].sort_values("val_pr_auc", ascending=False)
        chosen[family] = sub.iloc[0]["strategy"]
    print(f"  -> selected strategies: {chosen}")
    return table, {"chosen": chosen, "scale_pos_weight": spw}


# --------------------------------------------------------------------------- #
# Stage 4 — XGBoost hyperparameter search
# --------------------------------------------------------------------------- #
def tune_xgb(data: dict, strategy: str, spw: float) -> tuple[XGBClassifier, pd.DataFrame]:
    X_tr, y_tr = data["X_train"], data["y_train"]

    param_dist = {
        "n_estimators": [150, 200, 250, 300, 350, 400],
        "learning_rate": loguniform(0.01, 0.3),
        "max_depth": [3, 4, 5, 6, 7, 8],
        "min_child_weight": [1, 2, 3, 5, 7],
        "subsample": uniform(0.7, 0.3),
        "colsample_bytree": uniform(0.7, 0.3),
        "reg_lambda": [1.0, 1.5, 2.0, 3.0],
        "gamma": [0, 0.05, 0.1, 0.3],
    }

    base = xgb_classifier(strategy, spw=spw, n_estimators=300, n_jobs=1)
    if strategy == "smote":
        base = ImbPipeline([("smote", SMOTE(random_state=SEED)), ("clf", base)])
        param_dist = {f"clf__{k}": v for k, v in param_dist.items()}

    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=N_SEARCH_ITER,
        cv=StratifiedKFold(n_splits=SEARCH_CV, shuffle=True, random_state=SEED),
        scoring="average_precision",
        random_state=SEED,
        n_jobs=2,
        refit=True,
    )
    print(f"  RandomizedSearchCV: {N_SEARCH_ITER} candidates x {SEARCH_CV}-fold CV "
          f"(scoring = average_precision) ...")
    t0 = time.time()
    search.fit(X_tr, y_tr)
    print(f"  search finished in {time.time() - t0:.0f}s | best CV AP = "
          f"{search.best_score_:.4f}")

    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    results.to_csv(TUNING_DIR / "randomized_search_results.csv", index=False)

    best_params = dict(search.best_params_)
    print(f"  best params: {best_params}")
    if strategy == "smote":
        best_model = search.best_estimator_.named_steps["clf"]
        best_params = {k.replace("clf__", ""): v for k, v in best_params.items()}
    else:
        best_model = search.best_estimator_
    return best_model, best_params


# --------------------------------------------------------------------------- #
# Stage 5 — calibration selection (fit on train, select on val)
# --------------------------------------------------------------------------- #
def choose_calibration(best_xgb: XGBClassifier, data: dict) -> tuple[str, dict]:
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_va, y_va = data["X_val"], data["y_val"]

    comparison = {}
    for method in ["sigmoid", "isotonic"]:
        cal = CalibratedClassifierCV(best_xgb, method=method, cv=5)
        cal.fit(X_tr, y_tr)
        proba = cal.predict_proba(X_va)[:, 1]
        comparison[method] = {
            "brier": round(brier_score_loss(y_va, proba), 4),
            "roc_auc": round(roc_auc_score(y_va, proba), 4),
            "pr_auc": round(average_precision_score(y_va, proba), 4),
        }
        print(f"  calibration={method:9s} | val Brier {comparison[method]['brier']:.4f} "
              f"| ROC-AUC {comparison[method]['roc_auc']:.4f}")

    best_method = min(comparison, key=lambda m: comparison[m]["brier"])
    print(f"  -> selected calibration: {best_method}")
    return best_method, comparison


# --------------------------------------------------------------------------- #
# Stage 6 — final models (train+val)
# --------------------------------------------------------------------------- #
def fit_final_models(data: dict, xgb_strategy: str, spw: float, best_params: dict,
                     calib_method: str) -> dict:
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_va, y_va = data["X_val"], data["y_val"]
    X_full = pd.concat([X_tr, X_va], axis=0)
    y_full = pd.concat([y_tr, y_va], axis=0)

    models = {}

    print("  [final] dummy (prior) ...")
    models["dummy"] = DummyClassifier(strategy="prior").fit(X_full, y_full)

    print("  [final] logistic regression ...")
    models["logistic"] = lr_pipeline("none", data["numeric_cols"]).fit(X_full, y_full)

    print("  [final] decision tree ...")
    models["decision_tree"] = DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=50, random_state=SEED
    ).fit(X_full, y_full)

    print("  [final] random forest ...")
    models["random_forest"] = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        random_state=SEED, n_jobs=2,
    ).fit(X_full, y_full)

    print("  [final] xgboost (tuned) ...")
    xgb_final = xgb_classifier(xgb_strategy, spw=spw, n_jobs=2, **best_params)
    if xgb_strategy == "smote":
        xgb_final = ImbPipeline(
            [("smote", SMOTE(random_state=SEED)), ("clf", xgb_final)]
        ).fit(X_full, y_full)
    else:
        xgb_final.fit(X_full, y_full)
    models["xgboost"] = xgb_final

    print(f"  [final] xgboost calibrated ({calib_method}) ...")
    cal_base = xgb_classifier(xgb_strategy, spw=spw, n_jobs=2, **best_params)
    if xgb_strategy == "smote":
        cal_base = ImbPipeline(
            [("smote", SMOTE(random_state=SEED)), ("clf", cal_base)]
        )
    models["xgboost_calibrated"] = CalibratedClassifierCV(
        cal_base, method=calib_method, cv=5
    ).fit(X_full, y_full)

    for name, model in models.items():
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, path, compress=3)
        print(f"    saved -> {path}")
    return models


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> dict:
    t_start = time.time()
    print("[train] preparing feature matrices ...")
    data = prepare_matrices()
    print(f"  features: {len(data['feature_cols'])} "
          f"({len(data['cat_cols'])} categorical codes + "
          f"{len(data['numeric_cols'])} numeric)")

    print("[train] stage 3 — imbalance handling experiments (validation) ...")
    imb_table, imb_info = run_imbalance_experiments(data)
    lr_strategy = imb_info["chosen"]["logistic"]
    xgb_strategy = imb_info["chosen"]["xgboost"]
    spw = imb_info["scale_pos_weight"]

    print(f"[train] stage 4 — XGBoost tuning (strategy = {xgb_strategy}) ...")
    best_xgb, best_params = tune_xgb(data, xgb_strategy, spw)
    best_params = {
        k: float(v) if isinstance(v, (float, np.floating))
        else int(v) if isinstance(v, (int, np.integer)) else v
        for k, v in best_params.items()
    }

    print("[train] stage 5 — probability calibration selection ...")
    calib_method, calib_comparison = choose_calibration(best_xgb, data)

    print("[train] stage 6 — refit final models on train+val ...")
    fit_final_models(data, xgb_strategy, spw, best_params, calib_method)

    metadata = {
        "feature_cols": data["feature_cols"],
        "cat_cols": data["cat_cols"],
        "numeric_cols": data["numeric_cols"],
        "imbalance": {
            "selected_strategy_xgb": xgb_strategy,
            "selected_strategy_lr": lr_strategy,
            "scale_pos_weight": round(spw, 4),
            "experiments": imb_table.to_dict(orient="records"),
        },
        "xgb_best_params": best_params,
        "calibration": {"method": calib_method, "comparison": calib_comparison},
        "search": {"n_iter": N_SEARCH_ITER, "cv": SEARCH_CV,
                   "scoring": "average_precision"},
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[train] done in {metadata['runtime_seconds']}s -> models/metadata.json")
    return metadata


if __name__ == "__main__":
    main()
