"""
train_models.py - train the multi-class models on a topology's dataset and save its Stage-2 bundle.

Data, collected on the live emulated network by routing/collect_dataset.py:
    geant2   data/geant2_dataset.csv   (500 paths, 125 per class)
    fattree  data/fattree_dataset.csv  (896 paths, 224 per class; --topology fattree)
Classes: 0 Extremely Low, 1 Low, 2 High, 3 Extremely High. Split 70 / 15 / 15 (stratified, seed 40).

Models: XGBoost, Random Forest, Gradient Boosting, SVM (RBF, calibrated probabilities), and their
soft-voting ensemble (mean of the four probability vectors). All are saved in one bundle,
models/<topology>_bundle.joblib; Stage 2 uses the ensemble.

Outputs in results/training/<topology>/: model_comparison.csv, confusion_matrix_<model>.csv,
model_comparison.png.

    .venv/bin/python routing/train_models.py
    .venv/bin/python routing/train_models.py --topology fattree
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from telemetry_and_model import CLASS_NAMES, FEATURE_NAMES, MODEL_PATHS, SoftVotingEnsemble  # noqa: E402

log = logging.getLogger("train_models")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = {t: os.path.join(PROJECT_DIR, "data", f"{t}_dataset.csv") for t in ("geant2", "fattree")}
RESULTS = {t: os.path.join(PROJECT_DIR, "results", "training", t) for t in ("geant2", "fattree")}
SEED = 40


def build_models() -> dict:
    return {
        "XGBoost": xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", early_stopping_rounds=10,
                                     n_estimators=500, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "SVM": make_pipeline(StandardScaler(), CalibratedClassifierCV(
            SVC(kernel="rbf", class_weight="balanced", random_state=SEED), ensemble=False)),
    }


def scores(y_true, y_pred) -> dict:
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0)}


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path).dropna(subset=FEATURE_NAMES + ["traffic_class"])
    y = df["traffic_class"].astype(int).to_numpy()
    if sorted(set(y)) != list(range(len(CLASS_NAMES))):
        raise ValueError(f"Dataset must contain all 4 classes, has {sorted(set(y))}")
    return df[FEATURE_NAMES].to_numpy(dtype=float), y


def plot_comparison(table: pd.DataFrame, path: str, test_size: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualize_benchmarks import INK, INK_2, MUTED, apply_style

    apply_style()
    order = table["model"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.5 * len(order) + 1.4), sharey=True)
    for ax, (column, title) in zip(axes, [("accuracy", "Accuracy"), ("precision", "Precision (macro)"),
                                          ("f1", "F1 (macro)")]):
        values = table.set_index("model").loc[order, column]
        colors = ["#2a78d6" if m == "Ensemble (soft vote)" else "#c3c2b7" for m in order]
        ax.barh(order, values, color=colors, height=0.62)
        for y, value in enumerate(values):
            ax.annotate(f"{value:.3f}", xy=(value, y), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK if order[y].startswith("Ensemble") else INK_2)
        ax.set_xlim(0, 1.12)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    fig.suptitle(f"Test-set scores ({test_size} paths); blue: the soft-voting ensemble used by Stage 2", x=0.01, ha="left",
                 fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def train(args) -> pd.DataFrame:
    X, y = load_dataset(args.data)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)
    log.info("%d paths, %d features; train %d / validation %d / test %d", len(y), len(FEATURE_NAMES),
             len(y_train), len(y_val), len(y_test))
    fitted, rows = {}, []
    for name, model in build_models().items():
        start = time.perf_counter()
        if name == "XGBoost":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)
        fitted[name] = model
        rows.append((name, model, time.perf_counter() - start))
    ensemble = SoftVotingEnsemble(fitted)
    rows.append(("Ensemble (soft vote)", ensemble, sum(r[2] for r in rows)))

    table, matrices = [], {}
    for name, model, train_s in rows:
        start = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_ms = (time.perf_counter() - start) * 1000 / len(X_test)
        table.append({"model": name, **scores(y_test, y_pred),
                      "val_f1": f1_score(y_val, model.predict(X_val), average="macro"),
                      "train_s": train_s, "infer_ms_per_path": infer_ms})
        matrices[name] = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_NAMES))))
    table = pd.DataFrame(table)

    print(f"\n{'Model':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}{'Val F1':>9}{'ms/path':>9}")
    for _, r in table.iterrows():
        print(f"{r['model']:<22}{r['accuracy']:>10.3f}{r['precision']:>11.3f}{r['recall']:>9.3f}{r['f1']:>8.3f}"
              f"{r['val_f1']:>9.3f}{r['infer_ms_per_path']:>9.3f}")
    print("\nEnsemble confusion matrix on the test set (rows = actual, columns = predicted):")
    print(pd.DataFrame(matrices["Ensemble (soft vote)"], index=CLASS_NAMES, columns=CLASS_NAMES).to_string())

    os.makedirs(args.results, exist_ok=True)
    table.round(6).to_csv(os.path.join(args.results, "model_comparison.csv"), index=False)
    for name, matrix in matrices.items():
        slug = name.lower().split(" (")[0].replace(" ", "_")
        pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            os.path.join(args.results, f"confusion_matrix_{slug}.csv"))
    plot_comparison(table, os.path.join(args.results, "model_comparison.png"), len(y_test))

    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)
    joblib.dump({"models": fitted, "ensemble": ensemble, "features": FEATURE_NAMES, "classes": CLASS_NAMES,
                 "metrics": table.set_index("model").round(4).to_dict(orient="index"),
                 "dataset": os.path.relpath(args.data, PROJECT_DIR), "topology": args.topology}, args.model_out)
    print(f"\nSaved the bundle (4 models + ensemble) to {args.model_out}")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topology", default="geant2", choices=sorted(DATASETS),
                        help="sets the defaults of --data, --model-out and --results")
    parser.add_argument("--data")
    parser.add_argument("--model-out")
    parser.add_argument("--results")
    args = parser.parse_args()
    args.data = args.data or DATASETS[args.topology]
    args.model_out = args.model_out or MODEL_PATHS[args.topology]
    args.results = args.results or RESULTS[args.topology]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        train(args)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
