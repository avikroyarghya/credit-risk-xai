# Data Directory

## Layout

| Path | Content | Rules |
|---|---|---|
| `data/raw/` | Untouched UCI Excel download | **Never edited by hand.** Regenerated only by `src/data_loader.py`. |
| `data/processed/` | Cleaned dataset + stratified splits | Regenerated only by `src/preprocessing.py`. |

## Source

* **Dataset:** [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
  — UCI Machine Learning Repository (classification / business / credit risk).
* **Size:** 30,000 observations, 23 predictive features, binary target
  (`default payment next month`), no missing values reported.
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  (c) by I-Cheng Yeh.
* **Citation:** Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining
  techniques for the predictive accuracy of probability of default of credit card
  clients.* Expert Systems with Applications, 36(2), 2473–2480.

## Known Data-Quality Characteristics (audited)

A full independent audit of the raw file against the UCI documentation confirmed:
shape 30,000 × 24; all documented value ranges hold (`LIMIT_BAL` 10k–1M, `SEX` {1,2},
`EDUCATION` {0–6}, `MARRIAGE` {0–3}, `AGE` 21–79, `PAY_*` −2…9, `PAY_AMT*` ≥ 0);
0 missing cells; negative `BILL_AMT*` balances are legitimate (overpayment/credit).

| Finding | Count | Decision |
|---|---|---|
| Exact duplicate rows (all 24 columns) | 35 | Dropped in `preprocessing.py` → 29,965 clients |
| Undocumented `EDUCATION` codes 0/5/6 | 345 rows | Grouped into 4 (Others) |
| Undocumented `MARRIAGE` code 0 | 54 rows | Grouped into 3 (Others) |
| **Conflicting twin profiles** — identical 23-feature vectors with opposite targets | 21 profile pairs (42 rows) | **Kept** (see note) |

**Note on twin profiles.** After deduplication, 21 pairs of *distinct clients*
share an identical 23-feature history yet have opposite outcomes (one defaulted,
one did not). These are not data errors: they are separate customers whose
recorded histories coincide, and the label disagreement is genuine aleatoric
uncertainty — exactly what a probability model should average over. Nine of the
21 pairs happen to straddle split boundaries (5 train↔val, 4 train↔test); because
the labels conflict, the model cannot memorise these rows, so this does not
constitute target leakage. Impact is negligible (9 rows = 0.2% of the test set).

## Downloading

```bash
python -m src.data_loader      # downloads the .xls into data/raw/ (idempotent)
python -m src.preprocessing    # cleaning + 70/15/15 stratified splits
```

Mirrors are tried in order; the download is skipped when the raw file already
exists. Raw and processed data are **gitignored** — anyone reproducing the
project fetches the dataset themselves under the CC BY 4.0 terms.
