"""
profile_datasets.py - ydata-profiling (pandas profiling) reports of the two path datasets (report Section VII-C,
Figs. 3 and the dataset statistics).

Writes results/0_dataset_profiling/:
    <topology>/<topology>_profile.html, .json   the full profiling reports (the timestamp column is dropped)
    <topology>/charts/*.png                     the report's charts under readable names (histograms, correlation
                                                and missing-value diagrams)
    <topology>/alerts.txt                       the profiler's alerts
    feature_correlation_both.png / .csv         the profiler's "auto" correlation of the 11 features and the class,
                                                both datasets side by side (report Fig. 3)
    profiling_summary.csv                       rows, variables, missing cells, duplicate rows, alerts

    .venv/bin/python routing/profile_datasets.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from telemetry_and_model import FEATURE_NAMES  # noqa: E402
from visualize_benchmarks import INK, MUTED, SURFACE, apply_style  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT_DIR, "results", "0_dataset_profiling")
DATASETS = {"geant2": "GEANT2 path dataset (500 paths)", "fattree": "Fat-tree k=4 path dataset (896 paths)"}
TITLES = {"geant2": "GEANT2 (500 paths)", "fattree": "Fat-tree k=4 (896 paths)"}


def profile(topology: str, title: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ydata_profiling import ProfileReport
    df = pd.read_csv(os.path.join(PROJECT_DIR, "data", f"{topology}_dataset.csv"))
    df["traffic_class"] = df["traffic_class"].astype("category")
    df = df.drop(columns=["time"])          # wall-clock timestamp: not a property of the paths
    report = ProfileReport(df, title=title, explorative=True, progress_bar=False,
                           plot={"image_format": "png", "dpi": 200},
                           html={"inline": False, "assets_prefix": f"{topology}_assets"},
                           correlations={"pearson": {"calculate": True}, "spearman": {"calculate": True},
                                         "auto": {"calculate": True}},
                           interactions={"continuous": False})
    directory = os.path.join(OUT, topology)
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    html = os.path.join(directory, f"{topology}_profile.html")
    report.to_file(html)
    report.to_file(os.path.join(directory, f"{topology}_profile.json"))
    extract_charts(directory, html)
    data = json.load(open(os.path.join(directory, f"{topology}_profile.json")))
    with open(os.path.join(directory, "alerts.txt"), "w") as handle:
        handle.write("\n".join(data.get("alerts", [])))
    table = data["table"]
    return {"topology": topology, "rows": table["n"], "variables": table["n_var"],
            "missing_cells": table["n_cells_missing"], "duplicate_rows": table.get("n_duplicates"),
            "alerts": len(data.get("alerts", []))}


def extract_charts(directory: str, html_path: str) -> None:
    """Copy the report's PNG charts to charts/ under names derived from the HTML around each image."""
    html = open(html_path).read()
    charts = os.path.join(directory, "charts")
    os.makedirs(charts, exist_ok=True)
    seen = set()
    for match in re.finditer(r'<img([^>]*)src="?([^" >]+\.png)', html):
        attrs, src = match.group(1), match.group(2)
        alt = (re.search(r'alt="?([^">]+?)"?(?:\s|$)', attrs) or [None, ""])[1].strip()
        if "Mini" in alt:
            continue
        before = html[:match.start()]
        tail = before[-600:]
        if "pearson_diagram" in tail:
            name = "correlation_pearson"
        elif "spearman_diagram" in tail:
            name = "correlation_spearman"
        elif "auto_diagram" in tail:
            name = "correlation_auto"
        elif "missing-matrix" in tail[-400:] or "missing-bar" in tail[-400:]:
            name = "missing_values"
        else:
            var = re.findall(r'item-header h4" title=([^>]+)>', before)
            var = var[-1].strip('"') if var else "unknown"
            kind = "histogram" if alt == "Histogram" else ("length_histogram" if "length" in alt else "common_values")
            name = f"{var}_{kind}"
        base, i = name, 1
        while name in seen:
            i += 1
            name = f"{base}_{i}"
        seen.add(name)
        shutil.copyfile(os.path.join(directory, src), os.path.join(charts, f"{name}.png"))


def correlation_figure() -> None:
    """The profiler's 'auto' correlation of the features and the class, both datasets (report Fig. 3)."""
    columns = FEATURE_NAMES + ["traffic_class"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#c2410c", "#f0efec", "#2a78d6"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), gridspec_kw={"wspace": 0.55})
    frames = []
    for ax, topology in zip(axes, DATASETS):
        data = json.load(open(os.path.join(OUT, topology, f"{topology}_profile.json")))
        matrix = pd.DataFrame(data["correlations"]["auto"])
        matrix.index = matrix.columns
        matrix = matrix.loc[columns, columns]
        frames.append(matrix.assign(topology=topology))
        ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
        for i in range(len(columns)):
            for j in range(len(columns)):
                value = matrix.iat[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.6,
                        color=SURFACE if abs(value) > 0.6 else INK)
        ax.set_xticks(range(len(columns)), columns, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(columns)), columns, fontsize=7)
        ax.grid(False)
        ax.set_title(f"{TITLES[topology]}: feature correlation", pad=16)
        ax.text(0, 1.02, "ydata-profiling 'auto' correlation · traffic_class row/column: Cramér's V (unsigned)",
                transform=ax.transAxes, fontsize=7, color=MUTED)
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1)), ax=axes, shrink=0.7, label="correlation")
    fig.savefig(os.path.join(OUT, "feature_correlation_both.png"))
    plt.close(fig)
    pd.concat(frames).to_csv(os.path.join(OUT, "feature_correlation_both.csv"), float_format="%.4f")


def main() -> None:
    apply_style()
    os.makedirs(OUT, exist_ok=True)
    rows = [profile(topology, title) for topology, title in DATASETS.items()]
    apply_style()                           # the profiler resets matplotlib's style
    correlation_figure()
    with open(os.path.join(OUT, "profiling_summary.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
