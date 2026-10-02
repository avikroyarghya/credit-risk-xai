"""Data cleaning and stratified train/validation/test splitting.

Phase 2 (data understanding checks) + Phase 7 (leakage-safe splitting).

Decisions
---------
* Exact duplicate rows are dropped (35 found).
* EDUCATION codes 0/5/6 are undocumented in the UCI schema -> grouped
  into category 4 ("Others") instead of being silently kept.
* MARRIAGE code 0 is undocumented -> grouped into 3 ("Others").
* Splits are stratified on the target so the ~22% default rate is
  preserved in every partition. The test set stays untouched until the
  very end of the project.

Usage:
    python -m src.preprocessing
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import TARGET, load_raw, save_clean
from src.utils import PROCESSED_DIR, RANDOM_STATE

TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

# Undocumented category codes (per the UCI data description page).
EDUCATION_OTHERS = {0: 4, 5: 4, 6: 4}
MARRIAGE_OTHERS = {0: 3}

SPLIT_SIZES = {"train": 0.70, "val": 0.15, "test": 0.15}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented cleaning rules and return a tidy DataFrame."""
    df = df.copy()

    n_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # Group undocumented category codes into "Others".
    df["EDUCATION"] = df["EDUCATION"].replace(EDUCATION_OTHERS).astype(int)
    df["MARRIAGE"] = df["MARRIAGE"].replace(MARRIAGE_OTHERS).astype(int)

    # Logical-consistency guard: amounts cannot be negative in this schema
    # except that BILL_AMT can legitimately be negative (credit balance /
    # overpayment), so we do NOT clip bills. We only assert types.
    df = df.astype(int)

    print(f"[clean] dropped {n_dupes} duplicate rows -> {len(df):,} rows")
    print(f"[clean] EDUCATION values: {sorted(df['EDUCATION'].unique())}")
    print(f"[clean] MARRIAGE  values: {sorted(df['MARRIAGE'].unique())}")
    return df


def make_splits(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    seed: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    """Create stratified train/val/test splits (test_size implies the rest)."""
    test_size = round(1.0 - train_size - val_size, 10)

    train, rest = train_test_split(
        df,
        test_size=round(val_size + test_size, 10),
        stratify=df[TARGET],
        random_state=seed,
    )
    val, test = train_test_split(
        rest,
        test_size=round(test_size / (val_size + test_size), 10),
        stratify=rest[TARGET],
        random_state=seed,
    )
    return {"train": train, "val": val, "test": test}


def run_pipeline() -> pd.DataFrame:
    """raw -> clean csv -> stratified split files."""
    raw = load_raw()
    clean = clean_data(raw)
    save_clean(clean)

    splits = make_splits(clean)
    for name, part in splits.items():
        path = PROCESSED_DIR / f"{name}.csv"
        part.to_csv(path, index=False)
        print(
            f"[split] {name:5s}: {len(part):6,} rows "
            f"({len(part) / len(clean):.1%}) | default rate {part[TARGET].mean():.4f}"
        )
    return clean


def load_splits() -> dict[str, pd.DataFrame]:
    """Load the three split files (feature engineering is applied later)."""
    if not (TRAIN_CSV.exists() and VAL_CSV.exists() and TEST_CSV.exists()):
        run_pipeline()
    return {
        "train": pd.read_csv(TRAIN_CSV),
        "val": pd.read_csv(VAL_CSV),
        "test": pd.read_csv(TEST_CSV),
    }


if __name__ == "__main__":
    run_pipeline()
