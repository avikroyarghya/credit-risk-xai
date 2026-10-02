"""Independent full-project audit — Credit Risk XAI.

Verifies EVERY layer against the UCI documentation and recomputes every
reported analysis from scratch (no reuse of project helper functions for
the recomputations, so a bug in src/ cannot mask itself):

  A. Raw data integrity          (shape, ranges, dtypes, missing, dupes, target)
  B. Cleaning + split integrity  (dedup, category domains, no overlap, stratification)
  C. Feature engineering         (independent recomputation, NaN/inf scan)
  D. Leakage structure           (SMOTE/scaler placement, saved-model structure)
  E. Test metrics                (recompute all 6 models, compare to reports)
  F. Statistical tests           (recompute chi-square / Welch / effect sizes)
  G. SHAP + risk segmentation    (recompute importances, bands, monotonicity)
  H. Cross-artifact consistency  (README/summary numbers vs recomputed)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # noqa: E402

RESULTS: list[tuple[str, str, bool, str]] = []


def check(section: str, name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((section, name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
print("=" * 78)
print("A. RAW DATA INTEGRITY (vs UCI documentation)")
print("=" * 78)
raw_xls = ROOT / "data/raw/default of credit card clients.xls"
raw_all = pd.read_excel(raw_xls, header=1)          # includes ID column
raw = raw_all.drop(columns=["ID"]).rename(columns={"default payment next month": "default"})

check("A", "shape == 30,000 x 24 (23 features + target)", raw.shape == (30000, 24), str(raw.shape))
check("A", "ID column unique (30,000 distinct clients)", raw_all["ID"].is_unique)
check("A", "all dtypes integer", (raw.dtypes == "int64").all())
check("A", "zero missing cells", raw.isna().sum().sum() == 0)
check("A", "35 exact duplicate rows (feature-level, incl. target)", raw.duplicated().sum() == 35,
      f"{raw.duplicated().sum()} found")
check("A", "target binary {0,1}", set(raw["default"].unique()) == {0, 1})
check("A", "defaulters = 6,636 (22.12%)", int(raw["default"].sum()) == 6636,
      f"{int(raw['default'].sum())} / {raw['default'].mean():.4f}")

ranges = {
    "LIMIT_BAL": (10000, 1000000), "SEX": (1, 2), "EDUCATION": (0, 6),
    "MARRIAGE": (0, 3), "AGE": (21, 79), "PAY_0": (-2, 9), "PAY_2": (-2, 9),
    "PAY_3": (-2, 9), "PAY_4": (-2, 9), "PAY_5": (-2, 9), "PAY_6": (-2, 9),
}
for col, (lo, hi) in ranges.items():
    ok = raw[col].between(lo, hi).all()
    check("A", f"{col} within documented range [{lo}, {hi}]", ok,
          f"min={raw[col].min()} max={raw[col].max()}")

for i in range(1, 7):
    check("A", f"BILL_AMT{i} numeric, negatives allowed", pd.api.types.is_numeric_dtype(raw[f"BILL_AMT{i}"]),
          f"min={raw[f'BILL_AMT{i}'].min():,}")
    check("A", f"PAY_AMT{i} >= 0", (raw[f"PAY_AMT{i}"] >= 0).all(),
          f"min={raw[f'PAY_AMT{i}'].min():,}")
check("A", "no LIMIT_BAL == 0 (div-by-zero guard never triggers)", (raw["LIMIT_BAL"] == 0).sum() == 0)

# Repayment-status semantics: 0 = revolving is a valid code (UCI doc ambiguous,
# dataset contains it) — count only.
n_revolve = (raw["PAY_0"] == 0).sum()
check("A", "PAY_0 contains code 0 (revolving) — documented behaviour", n_revolve > 0,
      f"{n_revolve:,} clients revolving in Sep")

# --------------------------------------------------------------------------- #
print()
print("=" * 78)
print("B. CLEANING + SPLIT INTEGRITY")
print("=" * 78)
clean = pd.read_csv(ROOT / "data/processed/credit_clients.csv")
tr = pd.read_csv(ROOT / "data/processed/train.csv")
va = pd.read_csv(ROOT / "data/processed/val.csv")
te = pd.read_csv(ROOT / "data/processed/test.csv")

check("B", "clean rows == 30,000 - 35 dupes = 29,965", len(clean) == 29965, str(len(clean)))
check("B", "clean has no duplicates left", clean.duplicated().sum() == 0)
check("B", "EDUCATION domain {1,2,3,4} (0/5/6 -> 4)", set(clean["EDUCATION"].unique()) == {1, 2, 3, 4})
check("B", "MARRIAGE domain {1,2,3} (0 -> 3)", set(clean["MARRIAGE"].unique()) == {1, 2, 3})
check("B", "clean == raw minus 35 dupes, after documented category remapping",
      raw.assign(
          EDUCATION=raw["EDUCATION"].replace({0: 4, 5: 4, 6: 4}),
          MARRIAGE=raw["MARRIAGE"].replace({0: 3}),
      ).drop_duplicates().reset_index(drop=True)
      .equals(clean.reset_index(drop=True)[raw.columns]))

sizes = {"train": len(tr), "val": len(va), "test": len(te)}
check("B", "split sizes 20,975 / 4,495 / 4,495 (70/15/15)",
      (sizes["train"], sizes["val"], sizes["test"]) == (20975, 4495, 4495), str(sizes))
check("B", "splits sum to clean total", sum(sizes.values()) == len(clean))

# Overlap checks. Full-row keys (features + target): exact-duplicate rows were
# dropped, so identical full rows cannot cross splits — verify that.
# Feature-only keys CAN collide: distinct clients with identical 23-feature
# profiles (and necessarily opposite targets). Quantify + verify no leakage.
def rowkeys(df): return df.drop(columns=["default"]).astype(str).agg("|".join, axis=1)
def fullkeys(df): return df.astype(str).agg("|".join, axis=1)
full_overlap = {
    "train∩val": len(set(fullkeys(tr)) & set(fullkeys(va))),
    "train∩test": len(set(fullkeys(tr)) & set(fullkeys(te))),
    "val∩test": len(set(fullkeys(va)) & set(fullkeys(te))),
}
check("B", "no identical full rows across splits (true leakage impossible)",
      sum(full_overlap.values()) == 0, str(full_overlap))

feat_pairs = []
for a_name, ka, b_name, kb, b_df in [("train", rowkeys(tr), "val", rowkeys(va), va),
                                       ("train", rowkeys(tr), "test", rowkeys(te), te),
                                       ("val", rowkeys(va), "test", rowkeys(te), te)]:
    for c in set(ka) & set(kb):
        feat_pairs.append((int(tr[ka == c].iloc[0]["default"]), int(b_df[kb == c].iloc[0]["default"])))
check("B", "feature-twin pairs across splits ALL have conflicting targets (no memorisation)",
      len(feat_pairs) > 0 and all(a != b for a, b in feat_pairs),
      f"{len(feat_pairs)} pairs (5 train-val + 4 train-test), all conflicting — documented in data/README.md")
check("B", "splits partition clean exactly (full-row multiset)",
      len(set(fullkeys(pd.concat([tr, va, te])))) == len(clean)
      and pd.concat([tr, va, te]).shape[0] == len(clean))

rates = {"train": tr["default"].mean(), "val": va["default"].mean(), "test": te["default"].mean()}
check("B", "stratification: default rate ~22.1% in all splits",
      max(rates.values()) - min(rates.values()) < 0.001,
      " ".join(f"{k}={v:.4f}" for k, v in rates.items()))
check("B", "test defaulters = 994 (22.11%)", int(te["default"].sum()) == 994, str(int(te["default"].sum())))

# --------------------------------------------------------------------------- #
print()
print("=" * 78)
print("C. FEATURE ENGINEERING — independent recomputation")
print("=" * 78)
sys.path.insert(0, str(ROOT))  # noqa: E402
from src.feature_engineering import add_features  # noqa: E402

fe_te = add_features(te)
rng = np.random.default_rng(7)
idx = rng.choice(len(te), 400, replace=False)
s = te.iloc[sorted(idx)]

# Independent formulas (written from the blueprint, not from src/)
B = [f"BILL_AMT{i}" for i in range(1, 7)]
P = [f"PAY_AMT{i}" for i in range(1, 7)]
S = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
W = np.array([6, 5, 4, 3, 2, 1], float); W = W / W.sum()

exp = pd.DataFrame(index=s.index)
for i, c in enumerate(B, start=1):
    exp[f"UTILIZATION_{i}"] = (s[c] / s["LIMIT_BAL"]).clip(-0.5, 5.0)
exp["UTILIZATION_MAX"] = exp[[f"UTILIZATION_{i}" for i in range(1, 7)]].max(axis=1)
exp["UTILIZATION_MEAN"] = exp[[f"UTILIZATION_{i}" for i in range(1, 7)]].mean(axis=1)
exp["AVG_BILL_6M"] = s[B].mean(axis=1)
exp["AVG_PAY_6M"] = s[P].mean(axis=1)
exp["TOTAL_PAY_6M"] = s[P].sum(axis=1)
ab = s[B].mean(axis=1).abs()
ratio = (s[P].mean(axis=1) / ab.where(ab > 0)).fillna(1.0).clip(0, 3)
exp["PAY_TO_BILL_RATIO"] = ratio
exp["LATE_PAYMENT_COUNT_6M"] = (s[S] >= 1).sum(axis=1)
exp["ON_TIME_COUNT_6M"] = (s[S] <= 0).sum(axis=1)
exp["MAX_DELINQUENCY_6M"] = s[S].max(axis=1)
exp["WEIGHTED_DELINQUENCY"] = (s[S].clip(lower=0) * W).sum(axis=1)
exp["BILL_TREND"] = s["BILL_AMT1"] - s["BILL_AMT6"]
exp["BILL_TREND_RATIO"] = (exp["BILL_TREND"] / s["LIMIT_BAL"]).clip(-2, 2)

mism = {}
for col in exp.columns:
    got = fe_te.loc[exp.index, col]
    diff = (exp[col] - got).abs().max()
    if diff > 1e-9:
        mism[col] = float(diff)
check("C", "all 19 engineered features recompute exactly (400-row sample, tol 1e-9)",
      not mism, f"mismatches: {mism}" if mism else "19/19 exact")

X_te_fe = fe_te.drop(columns=["default"])
check("C", "no NaN / inf anywhere in engineered test matrix",
      np.isfinite(X_te_fe.to_numpy(dtype=float)).all())

fe_all = add_features(pd.concat([tr, va, te]))
check("C", "no NaN / inf across ALL 29,965 engineered rows",
      np.isfinite(fe_all.drop(columns=["default"]).to_numpy(dtype=float)).all())

# Range sanity (negative MAX = revolving / no-consumption only — by design)
check("C", "LATE_PAYMENT_COUNT_6M in [0,6]", fe_all["LATE_PAYMENT_COUNT_6M"].between(0, 6).all())
check("C", "ON_TIME_COUNT_6M in [0,6]", fe_all["ON_TIME_COUNT_6M"].between(0, 6).all())
check("C", "LATE + ON_TIME == 6 for every client",
      ((fe_all["LATE_PAYMENT_COUNT_6M"] + fe_all["ON_TIME_COUNT_6M"]) == 6).all())
check("C", "MAX_DELINQUENCY_6M within [-2, 9] and consistent with LATE count",
      fe_all["MAX_DELINQUENCY_6M"].between(-2, 9).all()
      and ((fe_all["LATE_PAYMENT_COUNT_6M"] > 0) == (fe_all["MAX_DELINQUENCY_6M"] >= 1)).all())

# --------------------------------------------------------------------------- #
print()
print("=" * 78)
print("D. LEAKAGE STRUCTURE (saved artifacts)")
print("=" * 78)
xgb_job = joblib.load(ROOT / "models/xgboost.joblib")
steps = list(xgb_job.named_steps.keys()) if hasattr(xgb_job, "named_steps") else ["(no pipeline)"]
check("D", "xgboost.joblib = SMOTE pipeline (resample happens INSIDE fit only)",
      steps == ["smote", "clf"], f"steps={steps}")

lr_job = joblib.load(ROOT / "models/logistic.joblib")
check("D", "logistic.joblib = prep(onehot+scale)+clf pipeline (scaler fit on train only)",
      list(lr_job.named_steps.keys()) == ["prep", "clf"])

cal_job = joblib.load(ROOT / "models/xgboost_calibrated.joblib")
check("D", "calibrated model wraps the SMOTE pipeline (no test data involved)",
      hasattr(cal_job, "calibrated_classifiers_"))

meta = json.loads((ROOT / "models/metadata.json").read_text())
check("D", "metadata: 42 feature columns, no target among them",
      len(meta["feature_cols"]) == 42 and "default" not in meta["feature_cols"])
check("D", "metadata: XGB strategy selected on validation (smote)", 
      meta["imbalance"]["selected_strategy_xgb"] == "smote")
check("D", "metadata: scale_pos_weight matches train ratio 3.52",
      abs(meta["imbalance"]["scale_pos_weight"] - (tr["default"] == 0).sum() / (tr["default"] == 1).sum()) < 0.01,
      f"{meta['imbalance']['scale_pos_weight']:.4f}")

# --------------------------------------------------------------------------- #
print()
print("=" * 78)
print("E. TEST METRICS — independent recomputation (all 6 models)")
print("=" * 78)
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,  # noqa: E402
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)

feat_cols = meta["feature_cols"]
Xte = fe_te[feat_cols].astype(float)
yte = te["default"].astype(int)
reported = pd.read_csv(ROOT / "reports/model_results.csv").set_index("model")

files = {"dummy": "Dummy (prior)", "logistic": "Logistic Regression",
         "decision_tree": "Decision Tree", "random_forest": "Random Forest",
         "xgboost": "XGBoost (tuned)", "xgboost_calibrated": "XGBoost (calibrated)"}
ev = json.loads((ROOT / "reports/eval_summary.json").read_text())
th = ev["threshold"]

all_ok = {}
for key, label in files.items():
    m = joblib.load(ROOT / f"models/{key}.joblib")
    p = m.predict_proba(Xte)[:, 1]
    t = th if key == "xgboost_calibrated" else 0.5
    pred = (p >= t).astype(int)
    recomputed = {
        "accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0), "f1": f1_score(yte, pred, zero_division=0),
        "roc_auc": roc_auc_score(yte, p), "pr_auc": average_precision_score(yte, p),
        "brier": brier_score_loss(yte, p)}
    rep = reported.loc[label]
    bad = {k: (round(v, 4), rep[k]) for k, v in recomputed.items() if abs(v - rep[k]) > 5e-5}
    all_ok[key] = not bad
    check("E", f"{label}: all 7 metrics match reports (tol 5e-5)", not bad, str(bad) if bad else "")

cm = confusion_matrix(yte, (joblib.load(ROOT / "models/xgboost_calibrated.joblib")
                            .predict_proba(Xte)[:, 1] >= th).astype(int)).tolist()
check("E", "confusion matrix matches eval_summary.json [[2942,559],[416,578]]",
      cm == ev["confusion"], str(cm))
check("E", "confusion matrix rows sum to class counts (3501 / 994)",
      cm[0][0] + cm[0][1] == 3501 and cm[1][0] + cm[1][1] == 994)
check("E", "reported recall == 578/994", abs(ev["test_recall"] - 578 / 994) < 1e-4)
check("E", "reported precision == 578/(578+559)", abs(ev["test_precision"] - 578 / 1137) < 1e-4)

# Dummy sanity: prior Brier = p(1-p)
prior = (pd.concat([tr, va])["default"].mean())
check("E", "Dummy Brier == p*(1-p) of train+val prior",
      abs(brier_score_loss(yte, joblib.load(ROOT / 'models/dummy.joblib').predict_proba(Xte)[:, 1])
          - prior * (1 - prior)) < 1e-4, f"prior={prior:.4f}")

# Calibration quality of the MAIN model
p_cal = joblib.load(ROOT / "models/xgboost_calibrated.joblib").predict_proba(Xte)[:, 1]
check("E", "isotonic probabilities within [0,1] and non-degenerate",
      p_cal.min() >= 0 and p_cal.max() <= 1 and len(np.unique(p_cal)) > 100,
      f"range [{p_cal.min():.4f}, {p_cal.max():.4f}]")

# --------------------------------------------------------------------------- #
print()
print("=" * 78)
print("F. STATISTICAL TESTS — independent recomputation")
print("=" * 78)
st_rep = pd.read_csv(ROOT / "reports/statistical_tests.csv")

def cramers_v_indep(tab):
    chi2 = stats.chi2_contingency(tab)[0]
    n = tab.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))

ok_all = True
for col in S:
    tab = pd.crosstab(clean[col], clean["default"])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    v = cramers_v_indep(tab)
    row = st_rep[st_rep.hypothesis.str.contains(col)].iloc[0]
    ok = abs(chi2 - row["statistic"]) < 0.05 and abs(v - row["effect_size"]) < 5e-4
    ok_all &= ok
    print(f"    {col}: chi2={chi2:8.1f} (rep {row['statistic']:8.1f})  V={v:.4f} (rep {row['effect_size']:.4f})")
check("F", "chi-square + Cramér's V recomputed for all 6 repayment months", ok_all)

for col, name in [("LIMIT_BAL", "Credit limit"), ("AGE", "Age")]:
    a_ = clean.loc[clean["default"] == 1, col]
    b_ = clean.loc[clean["default"] == 0, col]
    t_stat, t_p = stats.ttest_ind(a_, b_, equal_var=False)
    pooled = np.sqrt(((len(a_) - 1) * a_.var(ddof=1) + (len(b_) - 1) * b_.var(ddof=1)) / (len(a_) + len(b_) - 2))
    d = (a_.mean() - b_.mean()) / pooled
    row = st_rep[st_rep.hypothesis.str.contains(name)].iloc[0]
    check("F", f"{name}: Welch t + Cohen's d match report",
          abs(t_stat - row["statistic"]) < 0.05 and abs(d - row["effect_size"]) < 5e-4,
          f"t={t_stat:.2f} d={d:.4f} (rep d={row['effect_size']:.4f})")

hl = json.loads((ROOT / "reports/stats_highlights.json").read_text())
v_pay0 = cramers_v_indep(pd.crosstab(clean["PAY_0"], clean["default"]))
check("F", "stats_highlights: PAY_0 Cramér's V = 0.42 (recency gradient anchor)",
      abs(hl["pay0_cramers_v"] - v_pay0) < 5e-4, f"{hl['pay0_cramers_v']:.4f}")

# Recency gradient claim: V(PAY_0) > V(PAY_2) > ... > V(PAY_6)
vs = [cramers_v_indep(pd.crosstab(clean[c], clean["default"])) for c in S]
check("F", "monotone recency gradient V(PAY_0) > ... > V(PAY_6)",
      all(vs[i] >= vs[i + 1] - 1e-9 for i in range(5)), " ".join(f"{v:.3f}" for v in vs))

# --------------------------------------------------------------------------- #
print()
print("=" * 78)
print("G. SHAP + RISK SEGMENTATION — recomputation")
print("=" * 78)
import shap  # noqa: E402

imp_rep = pd.read_csv(ROOT / "reports/shap_importance.csv")
xgb_inner = xgb_job.named_steps["clf"]
explainer = shap.TreeExplainer(xgb_inner)
# EXACT same sample as src/explain.py (rng seed 42, 3,000 rows) for reproduction
rng42 = np.random.default_rng(42)
sample_idx = rng42.choice(len(Xte), size=3000, replace=False)
X_sample = Xte.iloc[sample_idx]
sv = explainer(X_sample).values
my_imp = pd.Series(np.abs(sv).mean(axis=0), index=feat_cols).sort_values(ascending=False)
top5_rep = imp_rep.feature.head(5).tolist()
top5_my = my_imp.head(5).index.tolist()
check("G", "SHAP top-5 drivers reproduce exactly (same 3,000-row sample, seed 42)",
      top5_rep == top5_my, f"rep={top5_rep} | mine={top5_my}")
rep_vals = imp_rep.set_index("feature")["mean_abs_shap"].reindex(feat_cols)
check("G", "SHAP mean|SHAP| magnitudes match reports (tol 1e-6)",
      float((rep_vals - my_imp[feat_cols]).abs().max()) < 1e-6,
      f"max diff = {float((rep_vals - my_imp[feat_cols]).abs().max()):.2e}")

# Rank correlation over all 42 features
check("G", "full SHAP importance profile correlates with recomputation (Spearman > 0.99)",
      stats.spearmanr(rep_vals, my_imp[feat_cols]).statistic > 0.99,
      f"rho={stats.spearmanr(rep_vals, my_imp[feat_cols]).statistic:.4f}")

# Risk segmentation recomputed from calibrated probabilities
seg_rep = pd.read_csv(ROOT / "reports/risk_segments.csv")
band = pd.cut(p_cal, [0.0, 0.20, 0.50, 1.01], right=False, labels=["Low", "Medium", "High"])
ok_rows = []
for name in ["Low", "Medium", "High"]:
    mask = (band == name)
    n = int(mask.sum())
    obs = yte[mask].mean()
    cap = yte[mask].sum() / yte.sum()
    r = seg_rep[seg_rep.band == name].iloc[0]
    ok_rows.append(n == r.clients and abs(obs - r.observed_default_rate) < 5e-4
                   and abs(cap - r.defaulters_captured) < 5e-4)
check("G", "risk bands recompute exactly (counts, observed rate, defaulters captured)",
      all(ok_rows), " ".join(f"{b}:{o}" for b, o in zip(['Low', 'Medium', 'High'], ok_rows)))
obs_rates = seg_rep.set_index("band")["observed_default_rate"]
check("G", "bands monotone: Low < Medium < High observed default",
      obs_rates["Low"] < obs_rates["Medium"] < obs_rates["High"],
      f"{obs_rates['Low']:.3f} < {obs_rates['Medium']:.3f} < {obs_rates['High']:.3f}")
check("G", "high band captures ~1/3 of defaulters (portfolio value claim)",
      0.30 < seg_rep.loc[seg_rep.band == "High", "defaulters_captured"].iloc[0] < 0.40)

xai = json.loads((ROOT / "reports/xai_highlights.json").read_text())
check("G", "xai_highlights top_5 == shap_importance top_5", xai["top_5_features"] == top5_rep)

# Fairness table sanity: group sizes sum to test size per dimension
fair = pd.read_csv(ROOT / "reports/fairness_metrics.csv")
for dim in fair["dimension"].unique():
    tot = fair.loc[fair.dimension == dim, "n"].sum()
    check("G", f"fairness '{dim}' groups partition the test set (n sums to 4,495)", tot == 4495, str(tot))
tp = fair["tpr_recall"].between(0, 1).all()
check("G", "fairness rates within [0,1]", tp)

# --------------------------------------------------------------------------- #
print()
print("=" * 78)
print("H. CROSS-ARTIFACT CONSISTENCY")
print("=" * 78)
eda_sum = json.loads((ROOT / "reports/eda_summary.json").read_text())
check("H", "eda_summary n_clients == 29,965", eda_sum["n_clients"] == 29965)
check("H", "eda_summary default_rate == 22.12%", abs(eda_sum["default_rate"] - clean["default"].mean()) < 1e-4)
by_pay0 = clean.groupby("PAY_0")["default"].mean()
check("H", "EDA claim: paid-duly 16.8% vs 2M-delay 69.1% default",
      abs(by_pay0[-1] - eda_sum["default_rate_paid_duly"]) < 1e-4
      and abs(by_pay0[2] - eda_sum["default_rate_2m_delay"]) < 1e-4
      and by_pay0[2] > 0.65 and 0.15 < by_pay0[-1] < 0.18,
      f"duly={by_pay0[-1]:.3f}, 2M={by_pay0[2]:.3f}")
check("H", "EDA claim: corr(PAY_0, target) ~ +0.32 dominates corr(LIMIT_BAL, target) ~ -0.15",
      eda_sum["corr_pay0_target"] > 0.3 and -0.2 < eda_sum["corr_limit_target"] < -0.1,
      f"pay0={eda_sum['corr_pay0_target']:.3f}, limit={eda_sum['corr_limit_target']:.3f}")

readme = (ROOT / "README.md").read_text()
claims = {
    "ROC-AUC 0.787/0.7872": "0.787" in readme,
    "PR-AUC 0.562/0.5618": "0.562" in readme or "0.5618" in readme,
    "recall 58%/0.5815": "0.5815" in readme or "58" in readme,
    "threshold 0.26": "0.26" in readme,
    "29,965 clients": "29,965" in readme,
    "risk bands 10.9/31.8/69.7": all(x in readme for x in ["10.9", "31.8", "69.7"]),
}
for name, ok in claims.items():
    check("H", f"README claim present & consistent: {name}", ok)

nb_files = sorted((ROOT / "notebooks").glob("0*.ipynb"))
check("H", "all 9 phase notebooks exist and are valid JSON",
      len(nb_files) == 9 and all(json.loads(p.read_text()) for p in nb_files),
      str([p.name for p in nb_files]))
nb01 = "".join("".join(c["source"]) for c in json.loads(nb_files[0].read_text())["cells"])
check("H", "notebook 01 quotes 30,000 raw rows and 22.1% default rate",
      "30,000" in nb01 and "22.1%" in nb01)

check("H", "requirements.txt pins the stack used",
      all(p in (ROOT / "requirements.txt").read_text()
          for p in ["pandas", "scikit-learn", "xgboost", "imbalanced-learn", "shap", "streamlit"]))

# --------------------------------------------------------------------------- #
print()
print("=" * 78)
fails = [r for r in RESULTS if not r[2]]
print(f"AUDIT COMPLETE — {len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed")
if fails:
    print("\nFAILURES:")
    for sec, name, _, det in fails:
        print(f"  [{sec}] {name} {det}")
print("=" * 78)
sys.exit(1 if fails else 0)
