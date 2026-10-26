"""
cross_validate.py - repeated stratified k-fold cross-validation of the four models and their soft-voting ensemble
(report Section V-C, "Why an ensemble rather than a single classifier").

For each of n_repeats x k folds the four models are trained exactly as in train_models.py (XGBoost early-stops
on a stratified 15 % validation split of the training folds) and scored on the held-out fold: macro-F1 and
multi-class log loss. Writes results/training/<topology>/cross_validation.csv (one row per fold and model) and
cross_validation_summary.csv (mean and standard deviation).

    .venv/bin/python routing/cross_validate.py                    # GEANT2, 10 x 5 folds
    .venv/bin/python routing/cross_validate.py --topology fattree
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from telemetry_and_model import SoftVotingEnsemble  # noqa: E402
from train_models import DATASETS, RESULTS, SEED, build_models, load_dataset  # noqa: E402


def cross_validate(topology: str, splits: int = 5, repeats: int = 10) -> pd.DataFrame:
    X, y = load_dataset(DATASETS[topology])
    rows = []
    folds = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=SEED)
    for fold, (train, test) in enumerate(folds.split(X, y)):
        X_fit, X_val, y_fit, y_val = train_test_split(X[train], y[train], test_size=0.15, random_state=SEED,
                                                      stratify=y[train])
        fitted = {}
        for name, model in build_models().items():
            if name == "XGBoost":
                model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X[train], y[train])
            fitted[name] = model
        for name, model in {**fitted, "Ensemble (soft vote)": SoftVotingEnsemble(fitted)}.items():
            proba = model.predict_proba(X[test])
            rows.append({"fold": fold, "model": name,
                         "macro_f1": f1_score(y[test], np.argmax(proba, axis=1), average="macro"),
                         "log_loss": log_loss(y[test], np.clip(proba, 1e-15, 1), labels=[0, 1, 2, 3])})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topology", default="geant2", choices=sorted(DATASETS))
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    table = cross_validate(args.topology, args.splits, args.repeats)
    summary = table.groupby("model")[["macro_f1", "log_loss"]].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    out = RESULTS[args.topology]
    os.makedirs(out, exist_ok=True)
    table.round(6).to_csv(os.path.join(out, "cross_validation.csv"), index=False)
    summary.round(4).to_csv(os.path.join(out, "cross_validation_summary.csv"))
    print(f"{args.topology}: {args.repeats} x {args.splits}-fold cross-validation")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
