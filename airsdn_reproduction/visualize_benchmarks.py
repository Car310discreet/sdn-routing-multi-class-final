"""
visualize_benchmarks.py - AIRSDN baseline figures: hop-count vs QoS-aware vs AIRSDN.

Reads airsdn_reproduction/results/runs/<run>/<approach>/{timeseries.csv,frames.csv,summary.json} and writes, per run,
into <run>/figures/:
    rtt_throughput.png   throughput and RTT over 60 s (paper Figs. 16-17)
    summary.png          mean RTT, mean throughput, PSNR, SSIM
    video_quality.png    per-frame PSNR and SSIM
    summary_table.csv

    .venv/bin/python airsdn_reproduction/visualize_benchmarks.py
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

log = logging.getLogger(__name__)

APPROACHES = ["hop_count", "qos_aware", "airsdn"]
LABELS = {"hop_count": "Hop-count", "qos_aware": "QoS-aware", "airsdn": "AIRSDN"}
# Categorical slots 1-3 of a colorblind-validated palette (light surface); aqua is below 3:1
# contrast, so every figure carries direct or value labels in ink colors.
COLORS = {"hop_count": "#2a78d6", "qos_aware": "#eb6834", "airsdn": "#1baf7a"}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "legend.frameon": False, "lines.linewidth": 2.0, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def load_run(run_dir: str) -> dict:
    data = {}
    for approach in APPROACHES:
        directory = os.path.join(run_dir, approach)
        if not os.path.isdir(directory):
            continue
        entry = {}
        for name in ("timeseries", "frames"):
            path = os.path.join(directory, f"{name}.csv")
            if os.path.exists(path):
                entry[name] = pd.read_csv(path)
        path = os.path.join(directory, "summary.json")
        if os.path.exists(path):
            entry["summary"] = json.load(open(path))
        data[approach] = entry
    if not data:
        raise ValueError(f"{run_dir}: no results")
    return data


def _end_labels(ax, ends: dict, x: float):
    """Direct labels at line ends, nudged apart so they never overlap."""
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    last = None
    for approach, y in sorted(ends.items(), key=lambda kv: kv[1]):
        if last is not None and y - last < 0.08 * span:
            y = last + 0.08 * span
        ax.annotate(LABELS[approach], xy=(x, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2)
        last = y


def plot_rtt_throughput(run, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
    for ax, (column, title) in zip(axes, [("throughput_mbps", "Throughput (Mbps)"), ("rtt_ms", "Round-trip time (ms)")]):
        ends = {}
        for approach, entry in run.items():
            ts = entry.get("timeseries")
            if ts is None:
                continue
            ax.plot(ts["t_s"], ts[column], color=COLORS[approach], label=LABELS[approach])
            valid = ts[column].dropna()
            if len(valid):
                ends[approach] = valid.iloc[-1]
        ax.set_title(title)
        ax.set_xlim(0, 60)
        ax.set_ylim(bottom=0)
        _end_labels(ax, ends, 60)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("NSFNET: RTT and throughput on the selected route", x=0.01, ha="left", y=1.0,
                 fontsize=12, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)


def plot_summary(run, out_path):
    approaches = list(run)
    metrics = [("Mean RTT (ms)", "mean_rtt_ms", "{:.0f}"), ("Mean throughput (Mbps)", "mean_throughput_mbps", "{:.2f}"),
               ("PSNR (dB)", "psnr_db", "{:.2f}"), ("SSIM", "ssim", "{:.3f}")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    for ax, (title, key, fmt) in zip(axes, metrics):
        values = [run[a].get("summary", {}).get(key) for a in approaches]
        heights = [v if v is not None else 0 for v in values]
        bars = ax.bar(range(len(approaches)), heights, width=0.6, color=[COLORS[a] for a in approaches],
                      edgecolor=SURFACE, linewidth=2)
        for bar, value in zip(bars, values):
            ax.annotate("n/a" if value is None else fmt.format(value),
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)
        ax.set_xticks(range(len(approaches)), [LABELS[a] for a in approaches], fontsize=9)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        if key == "ssim":
            present = [v for v in values if v is not None]
            ax.set_ylim(max(0.0, min(present, default=0.9) - 0.05), 1.01)   # differences live near 1
        else:
            ax.set_ylim(0, max(heights + [1]) * 1.15)
    fig.suptitle("NSFNET: routing approaches compared", x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_video_quality(run, out_path, window=60):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (column, title) in zip(axes, [("psnr_db", f"Per-frame PSNR (dB, capped at 100, {window}-frame mean)"),
                                          ("ssim", f"Per-frame SSIM ({window}-frame mean)")]):
        for approach, entry in run.items():
            frames = entry.get("frames")
            if frames is None or column not in frames:
                continue
            values = frames[column].replace([np.inf, -np.inf], 100.0).clip(upper=100)  # identical frames: inf
            ax.plot(frames["frame"], values.rolling(window, min_periods=1).mean(), color=COLORS[approach],
                    label=LABELS[approach], linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Frame")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("NSFNET: received video quality", x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot the AIRSDN baseline comparison")
    parser.add_argument("results", nargs="?", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "runs"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    apply_style()
    runs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(args.results, "*", "run.json")))
    if not runs:
        sys.exit(f"No runs found in {args.results}")
    for run_dir in runs:
        try:
            run = load_run(run_dir)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            log.warning("Skipping %s: %s", run_dir, exc)
            continue
        fig_dir = os.path.join(run_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        for name, plot in (("rtt_throughput.png", plot_rtt_throughput), ("summary.png", plot_summary),
                           ("video_quality.png", plot_video_quality)):
            try:
                plot(run, os.path.join(fig_dir, name))
            except Exception as exc:
                log.warning("%s: %s failed: %s", run_dir, name, exc)
        pd.DataFrame([{"approach": LABELS[a], **{k: v for k, v in e.get("summary", {}).items() if k != "approach"}}
                      for a, e in run.items()]).to_csv(os.path.join(fig_dir, "summary_table.csv"), index=False)
        log.info("Figures written to %s", fig_dir)


if __name__ == "__main__":
    main()
