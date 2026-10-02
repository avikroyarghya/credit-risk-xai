"""Feature engineering (Phase 6).

Raw UCI variables -> behavioural risk features:

* Credit utilization      BILL_AMTi / LIMIT_BAL   (per month + aggregates)
* Average bill/payment    avg over the last 6 months
* Payment-to-bill ratio   avg_pay_6m / avg_bill_6m (guarded, clipped)
* Delinquency count       how many of the last 6 months had a delay
* Max delinquency         worst repayment status in the window
* Recency-weighted delay  recent months weigh more (PAY_0 is newest)
* Bill trend              BILL_AMT1 - BILL_AMT6 (growing debt?)

All features are computed row-wise from client history only, so they can
safely be applied to any single customer at inference time (the Streamlit
predictor reuses this exact function).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import TARGET

BILL_COLS = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
PAYAMT_COLS = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]  # newest -> oldest

# Recency weights for PAY_0 (Sep) ... PAY_6 (Apr): newest month matters most.
RECENCY_WEIGHTS = np.array([6, 5, 4, 3, 2, 1], dtype=float)
RECENCY_WEIGHTS = RECENCY_WEIGHTS / RECENCY_WEIGHTS.sum()

CATEGORICAL_RAW = ["SEX", "EDUCATION", "MARRIAGE"]
DROP_AT_MODELING = ["SEX_LABEL", "EDUCATION_LABEL", "MARRIAGE_LABEL"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with engineered features appended."""
    out = df.copy()
    limit = out["LIMIT_BAL"].replace(0, np.nan)

    # --- Credit utilization (per month + aggregates) ---------------------- #
    for i, col in enumerate(BILL_COLS, start=1):
        out[f"UTILIZATION_{i}"] = (out[col] / limit).clip(-0.5, 5.0)
    out["UTILIZATION_MAX"] = out[[f"UTILIZATION_{i}" for i in range(1, 7)]].max(axis=1)
    out["UTILIZATION_MEAN"] = out[[f"UTILIZATION_{i}" for i in range(1, 7)]].mean(axis=1)
    out["UTILIZATION_LATEST"] = out["UTILIZATION_1"]

    # --- Averages ---------------------------------------------------------- #
    out["AVG_BILL_6M"] = out[BILL_COLS].mean(axis=1)
    out["AVG_PAY_6M"] = out[PAYAMT_COLS].mean(axis=1)
    out["TOTAL_PAY_6M"] = out[PAYAMT_COLS].sum(axis=1)

    # --- Payment-to-bill ratio (guarded against zero bills) --------------- #
    avg_bill = out["AVG_BILL_6M"].abs()
    ratio = out["AVG_PAY_6M"] / avg_bill.where(avg_bill > 0)
    out["PAY_TO_BILL_RATIO"] = ratio.fillna(1.0).clip(0.0, 3.0)

    # --- Delinquency behaviour --------------------------------------------- #
    status = out[PAY_STATUS_COLS]
    out["LATE_PAYMENT_COUNT_6M"] = (status >= 1).sum(axis=1)
    out["ON_TIME_COUNT_6M"] = (status <= 0).sum(axis=1)
    out["MAX_DELINQUENCY_6M"] = status.max(axis=1)
    # Recency-weighted severity: delays in recent months count more.
    out["WEIGHTED_DELINQUENCY"] = (status.clip(lower=0) * RECENCY_WEIGHTS).sum(axis=1)

    # --- Debt trend -------------------------------------------------------- #
    out["BILL_TREND"] = out["BILL_AMT1"] - out["BILL_AMT6"]
    out["BILL_TREND_RATIO"] = (out["BILL_TREND"] / limit).clip(-2.0, 2.0)

    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Ordered model feature columns (everything except target/labels)."""
    drop = {TARGET, *DROP_AT_MODELING, "default"}
    return [c for c in df.columns if c not in drop]


def get_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Group feature names by role (used for preprocessing pipelines)."""
    cols = feature_columns(df)
    engineered = [
        c
        for c in cols
        if c.startswith(("UTILIZATION_", "AVG_", "TOTAL_", "PAY_TO_BILL", "LATE_", "ON_", "MAX_", "WEIGHTED_", "BILL_TREND"))
    ]
    bills = [c for c in cols if c in BILL_COLS]
    payamts = [c for c in cols if c in PAYAMT_COLS]
    paystatus = [c for c in cols if c in PAY_STATUS_COLS]
    known = set(engineered) | set(bills) | set(payamts) | set(paystatus) | set(CATEGORICAL_RAW) | {"LIMIT_BAL", "AGE"}
    other = [c for c in cols if c not in known]
    return {
        "categorical": CATEGORICAL_RAW,
        "ordinal_status": paystatus,
        "continuous_raw": ["LIMIT_BAL", "AGE", *bills, *payamts],
        "engineered": engineered,
        "other": other,
    }
