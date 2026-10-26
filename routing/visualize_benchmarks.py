"""
visualize_benchmarks.py - Module 5: benchmark figures for the combined architecture.

Reads results/runs/<run>/scenario_<k>/<paradigm>/{timeseries.csv, frames.csv, summary.json, decision.json}
and writes into <run>/figures/:
    timeseries_scenario_<k>.png   line plots: throughput, RTT and jitter per second, one figure per scenario
                                  (not averaged across scenarios, which would blur a congested one)
    packet_loss_boxplot.png       box plots: per-second packet loss of every paradigm, all scenarios pooled,
                                  with each scenario's whole-stream video loss as a dot
    video_quality.png             bar charts: PSNR and SSIM (mean over scenarios, dots = scenarios)
    computational_latency.png     bar charts: Stage 1 + Stage 2 compute time (ms) and the measurement time
                                  each decision needs (s): link telemetry window + Stage-2 probing
    summary_table.csv             per paradigm: means of every metric, chosen-path ground-truth class
    per_scenario.csv              every scenario x paradigm
With results/benchmark/decision_latency_benchmark_geant2.csv (run_experiment.py --benchmark-only) also:
    computational_latency_offline.png   Stage 1 time per method and K, plus Stage 2 inference

    .venv/bin/python routing/visualize_benchmarks.py results/runs           # every run
    .venv/bin/python routing/visualize_benchmarks.py results/runs/<run>
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

log = logging.getLogger("visualize_benchmarks")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PARADIGMS = ["hop_count", "latency", "bandwidth", "baseline", "proposed"]
LABELS = {"hop_count": "Hop-count", "latency": "Latency", "bandwidth": "Bandwidth (widest)",
          "baseline": "Baseline: hop-count + ML", "proposed": "Ours: combined + ML"}
SHORT = {"hop_count": "Hop-\ncount", "latency": "Latency", "bandwidth": "Band-\nwidth",
         "baseline": "Baseline\nhop + ML", "proposed": "Ours\ncombined\n+ ML"}
# Categorical slots 1-4 of a colour-blind-validated palette, kept per paradigm across every figure; bandwidth
# takes a deeper magenta step, which clears the all-pairs CVD check (dE >= 8) against the other four. Orange
# and yellow sit at dE 13.7 for normal vision and aqua / yellow / magenta are below 3:1 against the surface,
# so every bar carries a value label in ink, every line plot a legend and a per-paradigm line style.
COLORS = {"hop_count": "#2a78d6", "latency": "#eb6834", "bandwidth": "#b8407e", "baseline": "#1baf7a",
          "proposed": "#eda100"}
LINESTYLES = {"hop_count": "-", "latency": (0, (1, 1.5)), "bandwidth": (0, (6, 2)), "baseline": (0, (4, 2, 1, 2)),
              "proposed": "-"}
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


# ----------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------
def find_runs(path: str) -> list[str]:
    if glob.glob(os.path.join(path, "scenario_*")):
        return [path]
    runs = sorted(d for d in glob.glob(os.path.join(path, "*")) if glob.glob(os.path.join(d, "scenario_*")))
    if not runs:
        raise ValueError(f"{path}: no runs (scenario_* directories) found")
    return runs


def load_run(run_dir: str) -> dict:
    """{scenario: {paradigm: {"timeseries", "frames", "summary", "decision"}}}"""
    data = {}
    for scenario_dir in sorted(glob.glob(os.path.join(run_dir, "scenario_*")), key=lambda d: int(d.rsplit("_", 1)[1])):
        k = int(scenario_dir.rsplit("_", 1)[1])
        for paradigm in PARADIGMS:
            directory = os.path.join(scenario_dir, paradigm)
            if not os.path.exists(os.path.join(directory, "summary.json")):
                continue
            entry = {"summary": json.load(open(os.path.join(directory, "summary.json")))}
            for name in ("timeseries", "frames"):
                path = os.path.join(directory, f"{name}.csv")
                entry[name] = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
            path = os.path.join(directory, "decision.json")
            entry["decision"] = json.load(open(path)) if os.path.exists(path) else {}
            data.setdefault(k, {})[paradigm] = entry
    if not data:
        raise ValueError(f"{run_dir}: no results")
    return data


def per_scenario_table(data: dict) -> pd.DataFrame:
    rows = []
    for k, paradigms in data.items():
        for paradigm, entry in paradigms.items():
            s, d = entry["summary"], entry["decision"]
            timing = d.get("timing", {})
            truth = d.get("chosen_ground_truth") or {}
            rows.append({"scenario": k, "paradigm": paradigm, "psnr_db": s.get("psnr_db"), "ssim": s.get("ssim"),
                         "mean_rtt_ms": s.get("mean_rtt_ms"), "mean_jitter_ms": s.get("mean_jitter_ms"),
                         "mean_throughput_mbps": s.get("mean_throughput_mbps"),
                         "packet_loss_pct": s.get("packet_loss_pct"), "frames_decoded": s.get("frames_decoded"),
                         "chosen_path": "-".join(d.get("chosen_path") or []),
                         "chosen_true_class": truth.get("class"), "chosen_true_class_name": truth.get("class_name"),
                         "stage1_ms": timing.get("stage1_ms"), "stage2_inference_ms": timing.get("stage2_inference_ms"),
                         "telemetry_s": timing.get("telemetry_s"), "stage2_probe_s": timing.get("stage2_probe_s")})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------
def plot_timeseries(paradigms: dict, scenario: int, path: str, title_prefix: str = "") -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.2), sharex=True)
    panels = [("throughput_mbps", "Throughput (Mbit/s)"), ("rtt_ms", "RTT (ms)"), ("jitter_ms", "Jitter (ms)")]
    for ax, (column, title) in zip(axes, panels):
        for paradigm in PARADIGMS:
            ts = paradigms.get(paradigm, {}).get("timeseries")
            if ts is None or ts.empty or column not in ts:
                continue
            values = pd.to_numeric(ts[column], errors="coerce")
            style = {"marker": "o", "markersize": 3} if values.isna().any() else {}
            style.update(linestyle=LINESTYLES[paradigm], zorder=5 if paradigm == "proposed" else 3,
                         linewidth=2.4 if paradigm == "proposed" else 1.8)
            ax.plot(ts["t_s"], values, color=COLORS[paradigm], label=LABELS[paradigm], **style)
        if column == "rtt_ms":                 # a congested route's seconds-long RTT would flatten the rest
            rtts = pd.concat([pd.to_numeric(paradigms[p]["timeseries"][column], errors="coerce")
                              for p in paradigms if not paradigms[p]["timeseries"].empty]).dropna()
            if len(rtts) and rtts.max() > 20 * max(rtts.median(), 1e-3):
                ax.set_yscale("log")
                title += " · log scale"
        ax.set_title(title)
    axes[-1].set_xlabel("Time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), bbox_to_anchor=(0.5, 0.965),
               fontsize=8.5, handlelength=3)
    fig.suptitle(f"{title_prefix}Scenario {scenario}: QoS during the video transfer", x=0.01, y=0.995, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path)
    plt.close(fig)


def plot_loss_boxplot(data: dict, table: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    loss_box(ax, data, table, "Per-second packet loss (%), all scenarios pooled")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def loss_box(ax, data: dict, table: pd.DataFrame, title: str) -> None:
    present = [p for p in PARADIGMS if p in set(table["paradigm"])]
    series = []
    for paradigm in present:
        values = []
        for paradigms in data.values():
            ts = paradigms.get(paradigm, {}).get("timeseries")
            if ts is not None and not ts.empty:
                values += pd.to_numeric(ts["packet_loss_pct"], errors="coerce").dropna().tolist()
        series.append(values or [np.nan])
    box = ax.boxplot(series, positions=range(len(present)), widths=0.55, patch_artist=True, showfliers=True,
                     medianprops=dict(color=INK, linewidth=1.6), flierprops=dict(marker="o", markersize=3,
                                                                                  markerfacecolor=MUTED,
                                                                                  markeredgecolor="none"))
    for patch, paradigm in zip(box["boxes"], present):
        patch.set_facecolor(COLORS[paradigm])
        patch.set_edgecolor(SURFACE)
    for i, paradigm in enumerate(present):
        whole = table.loc[table["paradigm"] == paradigm, "packet_loss_pct"].astype(float)
        ax.scatter(np.full(len(whole), i + 0.33), whole, marker="D", s=26, color=SURFACE, edgecolors=INK,
                   linewidths=1.0, zorder=5, label="whole-stream video loss (per scenario)" if i == 0 else None)
        mean = np.nanmean(series[i]) if series[i] else np.nan
        ax.annotate(f"mean {mean:.1f}", xy=(i, 1.0), xycoords=("data", "axes fraction"), ha="center",
                    va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(present)), [SHORT[p] for p in present], fontsize=8.5)
    ax.set_title(title, pad=16)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.2), fontsize=8)     # below the ticks, off the boxes
    ax.grid(axis="x", visible=False)


def _bars(ax, table, key, title, hint, fmt):
    present = [p for p in PARADIGMS if p in set(table["paradigm"])]
    for i, paradigm in enumerate(present):
        values = pd.to_numeric(table.loc[table["paradigm"] == paradigm, key], errors="coerce").dropna()
        mean = values.mean() if len(values) else 0.0
        ax.bar(i, mean, width=0.62, color=COLORS[paradigm], zorder=2)
        if len(values) > 1:
            ax.vlines(i, values.min(), values.max(), color=INK_2, linewidth=1.2, zorder=3)
        ax.scatter(np.full(len(values), i), values, s=22, color=SURFACE, edgecolors=INK, linewidths=1.0, zorder=4)
        ax.annotate("n/a" if not len(values) else fmt.format(mean), xy=(i, max(mean, values.max() if len(values) else 0)),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(present)), [SHORT[p] for p in present], fontsize=8)
    ax.set_title(title, pad=16)
    ax.text(0, 1.02, hint, transform=ax.transAxes, fontsize=7.6, color=MUTED, va="bottom")
    top = pd.to_numeric(table[key], errors="coerce").max()
    ax.set_ylim(0, (top if np.isfinite(top) and top > 0 else 1) * 1.22)
    ax.grid(axis="x", visible=False)


def plot_video_quality(table: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _bars(axes[0], table, "psnr_db", "PSNR (dB)", "higher is better · bar = mean, dots = scenarios", "{:.1f}")
    _bars(axes[1], table, "ssim", "SSIM", "higher is better · bar = mean, dots = scenarios", "{:.3f}")
    fig.tight_layout(w_pad=2)
    fig.savefig(path)
    plt.close(fig)


def plot_computational_latency(table: pd.DataFrame, path: str) -> None:
    present = [p for p in PARADIGMS if p in set(table["paradigm"])]
    means = table.groupby("paradigm")[["stage1_ms", "stage2_inference_ms", "telemetry_s", "stage2_probe_s"]].mean()
    means = means.reindex(present).fillna(0.0)
    x = np.arange(len(present))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ax.bar(x, means["stage1_ms"], width=0.6, color=[COLORS[p] for p in present], zorder=2, label="Stage 1 (path discovery)")
    ax.bar(x, means["stage2_inference_ms"], bottom=means["stage1_ms"], width=0.6, zorder=2,
           color=[COLORS[p] for p in present], alpha=0.45, label="Stage 2 (inference, all candidates)")
    for i, p in enumerate(present):
        total = means.loc[p, "stage1_ms"] + means.loc[p, "stage2_inference_ms"]
        ax.annotate(f"{total:.2f} ms", xy=(i, total), xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=8.5, color=INK)
    ax.set_xticks(x, [SHORT[p] for p in present], fontsize=8)
    ax.set_title("Decision compute time (ms)", pad=16)
    ax.text(0, 1.02, "solid: Stage 1 · light: Stage 2 inference · mean over scenarios", transform=ax.transAxes,
            fontsize=7.6, color=MUTED, va="bottom")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    ax.bar(x, means["telemetry_s"], width=0.6, color="#c3c2b7", zorder=2, label="link telemetry window")
    ax.bar(x, means["stage2_probe_s"], bottom=means["telemetry_s"], width=0.6, zorder=2,
           color=[COLORS[p] for p in present], label="Stage 2 active probing")
    for i, p in enumerate(present):
        total = means.loc[p, "telemetry_s"] + means.loc[p, "stage2_probe_s"]
        ax.annotate(f"{total:.0f} s", xy=(i, total), xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=8.5, color=INK)
    ax.set_xticks(x, [SHORT[p] for p in present], fontsize=8)
    ax.set_title("Measurement time per decision (s)", pad=16)
    ax.text(0, 1.02, "grey: link telemetry window · colour: Stage-2 probing of the candidates",
            transform=ax.transAxes, fontsize=7.6, color=MUTED, va="bottom")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=3)
    fig.savefig(path)
    plt.close(fig)


def plot_offline_latency(csv_path: str, path: str, topology: str = "GEANT2") -> None:
    """Stage 1 per method and K (left) and Stage 2 batched inference per K (right): separate panels, because
    inference is an order of magnitude larger and would hide the differences between the methods."""
    df = pd.read_csv(csv_path)
    methods = ["hop", "latency", "bandwidth", "composite"]
    names = ["Hop count", "Latency", "Available\nbandwidth", "Composite"]
    ks = sorted(df["k"].unique())
    shades = ["#86b6ef", "#2a78d6", "#184f95"]
    stage1 = df.groupby(["method", "k"])["stage1_ms"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    width = 0.8 / len(ks)
    for j, k in enumerate(ks):
        x = np.arange(len(methods)) + (j - (len(ks) - 1) / 2) * width
        values = [stage1.get((m, k), 0.0) for m in methods]
        ax.bar(x, values, width * 0.92, color=shades[j % len(shades)], zorder=2, label=f"K = {k}")
        for xi, v in zip(x, values):
            ax.annotate(f"{v:.2f}", xy=(xi, v), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8, color=INK)
    ax.set_xticks(range(len(methods)), names)
    ax.set_title("Stage 1: path discovery (ms)", pad=16)
    ax.text(0, 1.02, f"{topology}, synthetic link telemetry, mean of {df['trial'].nunique()} trials",
            transform=ax.transAxes, fontsize=7.6, color=MUTED, va="bottom")
    ax.set_ylim(0, max(stage1.max() * 1.25, 1))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    inference = df.groupby("k")["stage2_inference_ms"]
    means = inference.mean()
    ax.bar(range(len(ks)), [means[k] for k in ks], width=0.55, color=[shades[j % len(shades)] for j in range(len(ks))],
           zorder=2)
    for j, k in enumerate(ks):
        ax.annotate(f"{means[k]:.1f}", xy=(j, means[k]), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8, color=INK)
    ax.set_xticks(range(len(ks)), [f"K = {k}" for k in ks])
    ax.set_title("Stage 2: model inference (ms)", pad=16)
    ax.text(0, 1.02, "all K candidates in one batched call", transform=ax.transAxes, fontsize=7.6, color=MUTED,
            va="bottom")
    ax.set_ylim(0, means.max() * 1.25)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=3)
    fig.savefig(path)
    plt.close(fig)


def visualize(run_dir: str, benchmark_csv: str | None = None) -> pd.DataFrame:
    data = load_run(run_dir)
    table = per_scenario_table(data)
    out = os.path.join(run_dir, "figures")
    os.makedirs(out, exist_ok=True)
    apply_style()
    for k, paradigms in data.items():
        plot_timeseries(paradigms, k, os.path.join(out, f"timeseries_scenario_{k}.png"))
    plot_loss_boxplot(data, table, os.path.join(out, "packet_loss_boxplot.png"))
    plot_video_quality(table, os.path.join(out, "video_quality.png"))
    plot_computational_latency(table, os.path.join(out, "computational_latency.png"))
    if benchmark_csv and os.path.exists(benchmark_csv):
        plot_offline_latency(benchmark_csv, os.path.join(out, "computational_latency_offline.png"))
    table.to_csv(os.path.join(out, "per_scenario.csv"), index=False)
    numeric = ["psnr_db", "ssim", "mean_rtt_ms", "mean_jitter_ms", "mean_throughput_mbps", "packet_loss_pct",
               "frames_decoded", "chosen_true_class", "stage1_ms", "stage2_inference_ms", "telemetry_s",
               "stage2_probe_s"]
    summary = table.groupby("paradigm")[numeric].agg(lambda s: pd.to_numeric(s, errors="coerce").mean())
    summary = summary.reindex([p for p in PARADIGMS if p in summary.index])
    summary.insert(0, "scenarios", table.groupby("paradigm").size())
    summary.index = [LABELS[p] for p in summary.index]
    summary.round(4).to_csv(os.path.join(out, "summary_table.csv"))
    print(f"\n{run_dir}\n{summary.round(3).to_string()}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default=os.path.join(PROJECT_DIR, "results", "runs"))
    parser.add_argument("--benchmark", default=os.path.join(PROJECT_DIR, "results", "benchmark",
                                                            "decision_latency_benchmark_geant2.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        for run_dir in find_runs(args.path):
            visualize(run_dir, args.benchmark)
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
