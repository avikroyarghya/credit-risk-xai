"""One-command reproduction of the full project pipeline.

Usage (from the project root):
    python run_all.py            # everything
    python run_all.py --fast     # skip hyperparameter search (uses defaults)

Stages
------
1. data_loader        download UCI dataset (idempotent)
2. preprocessing      cleaning + stratified 70/15/15 split
3. eda                EDA figure set
4. statistical_analysis  hypothesis tests + effect sizes
5. train              models, imbalance, tuning, calibration
6. evaluate           test-set metrics, curves, threshold
7. explain            SHAP, fairness, risk segmentation
"""
from __future__ import annotations

import sys
import time

from src import data_loader, eda, evaluate, explain, preprocessing, statistical_analysis, train


def stage(title: str, fn, *args, **kwargs):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")
    t0 = time.time()
    fn(*args, **kwargs)
    print(f"[done] {title} ({time.time() - t0:.0f}s)")


def main() -> None:
    t_start = time.time()
    fast = "--fast" in sys.argv
    if fast:
        train.N_SEARCH_ITER = 3  # quick smoke run, not for reporting

    stage("Stage 1/7 — data acquisition", lambda: data_loader.load_raw())
    stage("Stage 2/7 — cleaning + stratified split", preprocessing.run_pipeline)
    stage("Stage 3/7 — exploratory data analysis", eda.run_all)
    stage("Stage 4/7 — statistical analysis", statistical_analysis.run_all)
    stage("Stage 5/7 — model training + tuning + calibration", train.main)
    stage("Stage 6/7 — test-set evaluation", evaluate.main)
    stage("Stage 7/7 — SHAP explainability + fairness + segmentation", explain.main)

    print(f"\n{'=' * 74}\nPipeline complete in {(time.time() - t_start) / 60:.1f} min")
    print("Launch the dashboard:  streamlit run app/app.py\n")


if __name__ == "__main__":
    main()
