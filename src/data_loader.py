"""Data acquisition for the UCI "Default of Credit Card Clients" dataset.

Phase 1 of the project pipeline:

    raw download -> data/raw/ (never manually edited) -> cleaned frame

The dataset (30,000 clients, 23 features + binary target) is published by
the UCI Machine Learning Repository under CC BY 4.0:

    Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining
    techniques for the predictive accuracy of probability of default of
    credit card clients. Expert Systems with Applications, 36(2), 2473-2480.

Usage (from the project root):

    python -m src.data_loader            # download (if needed) + summary
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.utils import PROCESSED_DIR, RAW_DIR

RAW_XLS = RAW_DIR / "default of credit card clients.xls"
RAW_CSV = RAW_DIR / "default_of_credit_card_clients.csv"
CLEAN_CSV = PROCESSED_DIR / "credit_clients.csv"

# Primary + fallback sources (both hosted by UCI).
RAW_URLS = [
    "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls",
]

TARGET = "default"

# Human-readable labels used across EDA / the dashboard.
SEX_LABELS = {1: "Male", 2: "Female"}
EDUCATION_LABELS = {
    1: "Graduate school",
    2: "University",
    3: "High school",
    4: "Others",
}
MARRIAGE_LABELS = {1: "Married", 2: "Single", 3: "Others"}

PAY_MONTHS = {
    "PAY_0": "Sep",
    "PAY_2": "Aug",
    "PAY_3": "Jul",
    "PAY_4": "Jun",
    "PAY_5": "May",
    "PAY_6": "Apr",
}
PAY_SCALE = {
    -2: "No consumption",
    -1: "Paid duly",
    0: "Revolving credit",
    1: "1M delay",
    2: "2M delay",
    3: "3M delay",
    4: "4M delay",
    5: "5M delay",
    6: "6M delay",
    7: "7M delay",
    8: "8M delay",
    9: "9M+ delay",
}


def download_raw(force: bool = False) -> Path:
    """Download the raw Excel file into data/raw/ (idempotent)."""
    if RAW_XLS.exists() and not force:
        return RAW_XLS

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for url in RAW_URLS:
        try:
            print(f"Downloading: {url}")
            tmp = RAW_DIR / "tmp_download"
            urllib.request.urlretrieve(url, tmp)
            if url.endswith(".zip"):
                with zipfile.ZipFile(tmp) as zf:
                    member = [n for n in zf.namelist() if n.lower().endswith(".xls")][0]
                    with zf.open(member) as src, open(RAW_XLS, "wb") as dst:
                        dst.write(src.read())
            else:
                RAW_XLS.write_bytes(tmp.read_bytes())
            tmp.unlink(missing_ok=True)
            if RAW_XLS.exists():
                print(f"Saved raw file -> {RAW_XLS}")
                return RAW_XLS
        except Exception as exc:  # noqa: BLE001 - keep trying mirrors
            print(f"  failed ({exc}); trying next mirror ...")
    raise RuntimeError("Could not download the dataset from any mirror.")


def load_raw() -> pd.DataFrame:
    """Read the raw Excel file into a tidy DataFrame.

    The UCI workbook stores the real header on the second sheet row, so we
    read with header=1, drop the ID key and rename the target column to
    `default`.
    """
    download_raw()
    df = pd.read_excel(RAW_XLS, header=1)
    if df.columns[0].upper() != "ID":  # defensive: header row shifted
        df = pd.read_excel(RAW_XLS, header=0)
    df = df.drop(columns=[c for c in df.columns if c.upper() == "ID"])
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": TARGET})
    df.columns = [c.strip() for c in df.columns]
    return df


def save_clean(df: pd.DataFrame) -> Path:
    """Persist the cleaned dataset (the analysis source of truth)."""
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    return CLEAN_CSV


def load_clean() -> pd.DataFrame:
    """Load the cleaned dataset (or build it from raw on first call)."""
    if CLEAN_CSV.exists():
        return pd.read_csv(CLEAN_CSV)
    from src.preprocessing import clean_data

    df = clean_data(load_raw())
    save_clean(df)
    return df


def data_report(df: pd.DataFrame) -> dict:
    """Basic data-understanding facts used in Phase 2."""
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "target_counts": df[TARGET].value_counts().to_dict(),
        "target_rate": float(df[TARGET].mean()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


if __name__ == "__main__":
    raw = load_raw()
    print("\n=== RAW DATA ===")
    print(f"shape          : {raw.shape}")
    print(f"columns        : {list(raw.columns)}")
    print(f"duplicates     : {raw.duplicated().sum()}")
    print(f"missing cells  : {raw.isna().sum().sum()}")
    print(f"default rate   : {raw[TARGET].mean():.4f}")
    print("\ndtypes:")
    print(raw.dtypes.value_counts().to_string())
    print("\nFirst rows:")
    print(raw.head(3).to_string())
    sys.exit(0)
