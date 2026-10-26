"""
final_results.py - collect every result of the GEANT2 vs fat-tree evaluation into results/, each figure next
to the data it was drawn from (the figures and tables of the report).

Inputs (no root, no network):
    a GEANT2 run and a fat-tree run of run_experiment.py   (default: the newest results/runs/geant2_* and
                                                            results/runs/fattree_*)
    results/benchmark/decision_latency_benchmark_geant2.csv   (run_experiment.py --benchmark-only)
    results/training/<topology>/                              (train_models.py)
Videos of a run whose scoring failed are re-scored first (multimedia_evaluator.rescore_run).

Output, in results/:
    README.md                    what each figure shows, its data file, the runs and settings used
    1_training/                  model comparison and confusion matrices of both topologies' ensembles
    2_path_selection/            throughput, RTT, jitter, PSNR, SSIM bars and the packet-loss box plot of the
                                 five methods, GEANT2 and fat-tree side by side
    3_video_timeseries/<topo>/   per-second throughput, RTT and jitter during the video, one figure per scenario
    4_overhead_geant2/           per-step decision overhead (proposed vs baseline vs hop-count) and batched vs
                                 per-candidate inference
    5_paths/<topo>/              the path every method chose in every scenario, on the topology map
(0_dataset_profiling/ is written by profile_datasets.py.)

    .venv/bin/python routing/final_results.py
    .venv/bin/python routing/final_results.py --geant2-run results/runs/geant2_<ts> --fattree-run results/runs/fattree_<ts>
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROUTING_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(ROUTING_DIR)
sys.path.insert(0, ROUTING_DIR)
import visualize_benchmarks as vb  # noqa: E402
from multimedia_evaluator import rescore_run  # noqa: E402
from topology_builder import CLASS_NAMES, build_topology, class_of, link_key  # noqa: E402
from visualize_benchmarks import COLORS, INK, INK_2, LABELS, MUTED, SHORT, SURFACE  # noqa: E402

logging.setLoggerClass(logging.Logger)   # importing Mininet replaces the logger class
log = logging.getLogger("final_results")

TOPOLOGIES = ["geant2", "fattree"]
TOPO_LABEL = {"geant2": "GEANT2", "fattree": "Fat-tree (k=4)"}
RUNS_DIR = os.path.join(PROJECT_DIR, "results", "runs")
BENCHMARK_DIR = os.path.join(PROJECT_DIR, "results", "benchmark")
TRAINING = {t: os.path.join(PROJECT_DIR, "results", "training", t) for t in ("geant2", "fattree")}
METRICS = [  # column, file name, title, hint, value format
    ("mean_throughput_mbps", "throughput", "Throughput (Mbit/s)", "higher is better", "{:.2f}"),
    ("mean_rtt_ms", "rtt", "RTT (ms)", "lower is better", "{:.0f}"),
    ("mean_jitter_ms", "jitter", "Jitter (ms)", "lower is better", "{:.1f}"),
    ("psnr_db", "psnr", "PSNR (dB)", "higher is better", "{:.1f}"),
    ("ssim", "ssim", "SSIM", "higher is better", "{:.3f}"),
]
OVERHEAD_PARADIGMS = ["hop_count", "baseline", "proposed"]
# (timing key, label, scale to the panel's unit) in pipeline order; ordinal blue ramp steps 250 -> 650
MEASUREMENT_STEPS = [("telemetry_s", "Link telemetry (LLDP latency + port counters)", 1.0),
                     ("probe_rules_ms", "Probe rule install / removal (REST)", 1e-3),
                     ("ping_s", "Ping, 50 echoes per candidate", 1.0),
                     ("iperf_s", "iperf3 TCP, 5 s per candidate", 1.0),
                     ("drain_s", "Queue drain after each probe", 1.0)]
COMPUTE_STEPS = [("stage1_ms", "Stage 1 path discovery", 1.0),
                 ("assembly_ms", "Feature-matrix assembly", 1.0),
                 ("inference_ms", "Stage 2 batched inference (all K)", 1.0),
                 ("select_ms", "Path selection", 1.0),
                 ("route_install_ms", "Route installation (REST)", 1.0)]
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
CLASS_COLORS = ["#e9cfa8", "#f0a85a", "#d9601f", "#962812"]     # ordinal: Extremely Low -> Extremely High


# ----------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------
def newest_run(topology: str) -> str | None:
    runs = sorted(d for d in glob.glob(os.path.join(RUNS_DIR, f"{topology}_*"))
                  if glob.glob(os.path.join(d, "scenario_*")))
    return runs[-1] if runs else None


def load(run_dir: str) -> tuple[dict, pd.DataFrame, dict]:
    data = vb.load_run(run_dir)
    table = vb.per_scenario_table(data)
    with open(os.path.join(run_dir, "run.json")) as handle:
        meta = json.load(handle)
    return data, table, meta


def present(table: pd.DataFrame) -> list[str]:
    return [p for p in vb.PARADIGMS if p in set(table["paradigm"])]


def write_csv(frame: pd.DataFrame, path: str) -> None:
    frame.to_csv(path, index=False, float_format="%.4f")


# ----------------------------------------------------------------------------------------
# 1. Training
# ----------------------------------------------------------------------------------------
def training(out: str) -> list[str]:
    from train_models import plot_comparison
    os.makedirs(out, exist_ok=True)
    notes, tables = [], []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, topology in zip(axes, TOPOLOGIES):
        source = TRAINING[topology]
        comparison = os.path.join(source, "model_comparison.csv")
        if not os.path.exists(comparison):
            ax.axis("off")
            notes.append(f"{TOPO_LABEL[topology]}: no training results in {source}")
            continue
        table = pd.read_csv(comparison)
        tables.append(table.assign(topology=topology))
        matrix = pd.read_csv(os.path.join(source, "confusion_matrix_ensemble.csv"), index_col=0)
        for name in os.listdir(source):
            if name.endswith(".csv"):
                shutil.copyfile(os.path.join(source, name), os.path.join(out, f"{topology}_{name}"))
        plot_comparison(table, os.path.join(out, f"{topology}_model_comparison.png"), int(matrix.to_numpy().sum()))
        values = matrix.to_numpy()
        ax.imshow(values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blue", [SURFACE, "#2a78d6"]))
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, str(values[i, j]), ha="center", va="center", fontsize=10,
                        color=SURFACE if values[i, j] > values.max() / 2 else INK)
        short = ["EL", "L", "H", "EH"]
        ax.set_xticks(range(4), short)
        ax.set_yticks(range(4), short)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("actual class")
        ax.grid(False)
        ensemble = table.set_index("model").loc["Ensemble (soft vote)"]
        ax.set_title(f"{TOPO_LABEL[topology]}: soft-voting ensemble", pad=16)
        ax.text(0, 1.02, f"test set {int(values.sum())} paths · accuracy {ensemble['accuracy']:.3f} · "
                         f"macro-F1 {ensemble['f1']:.3f}", transform=ax.transAxes, fontsize=7.6, color=MUTED)
    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(out, "ensemble_confusion_matrices.png"))
    plt.close(fig)
    if tables:
        write_csv(pd.concat(tables)[["topology"] + list(tables[0].columns.drop("topology"))],
                  os.path.join(out, "model_comparison_both.csv"))
    return notes


# ----------------------------------------------------------------------------------------
# 2. Path selection: QoS / QoE per paradigm
# ----------------------------------------------------------------------------------------
def path_selection(out: str, runs: dict) -> None:
    os.makedirs(out, exist_ok=True)
    per_scenario = pd.concat([table.assign(topology=t) for t, (_, table, _) in runs.items()])
    columns = ["topology", "scenario", "paradigm"] + [c for c in per_scenario.columns
                                                      if c not in ("topology", "scenario", "paradigm")]
    write_csv(per_scenario[columns], os.path.join(out, "per_scenario.csv"))
    numeric = [m[0] for m in METRICS] + ["packet_loss_pct", "frames_decoded", "chosen_true_class"]
    stats = per_scenario.groupby(["topology", "paradigm"])[numeric].agg(["mean", "std", "count"])
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]
    write_csv(stats.reset_index(), os.path.join(out, "summary_mean_std.csv"))

    for column, name, title, hint, fmt in METRICS:
        fig, axes = plt.subplots(1, len(runs), figsize=(5.6 * len(runs), 4.3), squeeze=False)
        for ax, (topology, (_, table, _)) in zip(axes[0], runs.items()):
            vb._bars(ax, table, column, f"{TOPO_LABEL[topology]}: {title}",
                     f"{hint} · bar = mean, dots = scenarios", fmt)
        fig.tight_layout(w_pad=3)
        fig.savefig(os.path.join(out, f"{name}.png"))
        plt.close(fig)

    fig, axes = plt.subplots(1, len(runs), figsize=(6.2 * len(runs), 4.4), squeeze=False)
    rows = []
    for ax, (topology, (data, table, _)) in zip(axes[0], runs.items()):
        vb.loss_box(ax, data, table, f"{TOPO_LABEL[topology]}: per-second packet loss (%)")
        for k, paradigms in data.items():
            for paradigm, entry in paradigms.items():
                ts = entry["timeseries"]
                if not ts.empty:
                    rows.append(ts[["t_s", "packet_loss_pct"]].assign(topology=topology, scenario=k, paradigm=paradigm))
    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(out, "packet_loss_boxplot.png"))
    plt.close(fig)
    if rows:
        write_csv(pd.concat(rows)[["topology", "scenario", "paradigm", "t_s", "packet_loss_pct"]],
                  os.path.join(out, "packet_loss_per_second.csv"))


# ----------------------------------------------------------------------------------------
# 3. Per-second QoS during the video
# ----------------------------------------------------------------------------------------
def video_timeseries(out: str, runs: dict) -> None:
    for topology, (data, _, _) in runs.items():
        directory = os.path.join(out, topology)
        os.makedirs(directory, exist_ok=True)
        rows = []
        for k, paradigms in data.items():
            vb.plot_timeseries(paradigms, k, os.path.join(directory, f"scenario_{k}.png"),
                               title_prefix=f"{TOPO_LABEL[topology]} · ")
            for paradigm, entry in paradigms.items():
                if not entry["timeseries"].empty:
                    rows.append(entry["timeseries"].assign(scenario=k, paradigm=paradigm))
        if rows:
            frame = pd.concat(rows)
            write_csv(frame[["scenario", "paradigm", "t_s", "throughput_mbps", "rtt_ms", "jitter_ms",
                             "packet_loss_pct"]], os.path.join(directory, "timeseries.csv"))


# ----------------------------------------------------------------------------------------
# 4. Overhead (GEANT2)
# ----------------------------------------------------------------------------------------
def overhead_steps(data: dict) -> pd.DataFrame:
    rows = []
    for k, paradigms in data.items():
        for paradigm in OVERHEAD_PARADIGMS:
            timing = paradigms.get(paradigm, {}).get("decision", {}).get("timing")
            if not timing:
                continue
            row = {"scenario": k, "paradigm": paradigm}
            for key in ("telemetry_s", "ping_s", "iperf_s", "drain_s", "stage1_ms", "assembly_ms", "select_ms",
                        "route_install_ms"):
                row[key] = float(timing.get(key) or 0.0)
            row["probe_rules_ms"] = float(timing.get("probe_install_ms") or 0) + float(timing.get("probe_remove_ms") or 0)
            row["inference_ms"] = float(timing.get("inference_ms", timing.get("stage2_inference_ms")) or 0.0)
            row["measurement_total_s"] = sum(row[k] * scale for k, _, scale in MEASUREMENT_STEPS)
            row["compute_total_ms"] = sum(row[k] for k, _, _ in COMPUTE_STEPS)
            rows.append(row)
    return pd.DataFrame(rows)


def _stacked(ax, means: pd.DataFrame, steps: list, unit: str, title: str, hint: str) -> None:
    """Stacked bar per paradigm (steps in pipeline order, bottom up) over a table of every step's mean, so
    steps too small to see as a segment still have their number; the table's row swatches are the legend."""
    order = [p for p in OVERHEAD_PARADIGMS if p in means.index]
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    top = max((sum(means.loc[p, k] * s for k, _, s in steps) for p in order), default=1.0) or 1.0
    cells = []
    for (key, label, scale), color in zip(steps, RAMP):
        values = np.array([means.loc[p, key] * scale for p in order])
        cells.append([f"{v:.3g}" if v < 100 else f"{v:.0f}" for v in values])
        ax.bar(x, values, bottom=bottom, width=0.58, color=color, edgecolor=SURFACE, linewidth=2, zorder=2)
        for i, v in enumerate(values):             # label segments tall enough to hold a number
            if v >= 0.07 * top:
                ax.text(x[i], bottom[i] + v / 2, f"{v:.1f}" if v < 100 else f"{v:.0f}", ha="center", va="center",
                        fontsize=7.6, color=SURFACE if color in RAMP[2:] else INK)
        bottom += values
    for i, total in enumerate(bottom):
        ax.annotate(f"{total:.2f} {unit}" if total < 10 else f"{total:.0f} {unit}", xy=(x[i], total), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xticks([])
    ax.set_ylim(0, top * 1.15)
    ax.set_title(title, pad=16)
    ax.text(0, 1.02, hint, transform=ax.transAxes, fontsize=7.6, color=MUTED, va="bottom")
    ax.grid(axis="x", visible=False)
    rows = len(steps) + 1
    swatches = RAMP[:len(steps)][::-1] + [SURFACE]
    table = ax.table(cellText=[[""] + r for r in cells[::-1]] + [[""] + [f"{t:.3g}" if t < 100 else f"{t:.0f}"
                                                                        for t in bottom]],
                     rowLabels=[label for _, label, _ in steps][::-1] + [f"Total ({unit})"],
                     colLabels=[""] + [LABELS[p] for p in order], cellLoc="center", loc="bottom",
                     bbox=[0, -0.075 * (rows + 1) - 0.02, 1, 0.075 * (rows + 1)])
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    cells_by_pos = table.get_celld()
    swatch_w = 0.035
    value_w = (1 - swatch_w) / len(order)
    for (r, c), cell in cells_by_pos.items():
        cell.set_edgecolor("#e1e0d9")
        cell.set_linewidth(0.6)
        cell.set_facecolor(SURFACE)
        if c == 0:
            cell.set_width(swatch_w)
            if r >= 1:
                cell.set_facecolor(swatches[r - 1])      # the row's bar colour: the legend
                cell.set_edgecolor(SURFACE)
                cell.set_linewidth(2)
        elif c > 0:
            cell.set_width(value_w)
        if c == -1:
            cell.set_text_props(ha="left", color=INK if r == rows else INK_2,
                                fontweight="bold" if r == rows else "normal")
            cell.PAD = 0.04
        elif r == 0 or r == rows:
            cell.set_text_props(color=INK, fontweight="bold")
    xs = 0.0
    for c in range(len(order) + 1):                # re-pack the columns after resizing them
        for r in range(rows + 1):
            cells_by_pos[r, c].set_x(xs)
        xs += cells_by_pos[0, c].get_width()


def overhead(out: str, data: dict | None, benchmark_csv: str) -> None:
    os.makedirs(out, exist_ok=True)
    if data is not None:
        steps = overhead_steps(data)
        write_csv(steps, os.path.join(out, "overhead_steps_per_scenario.csv"))
        keys = [k for k, _, _ in MEASUREMENT_STEPS + COMPUTE_STEPS] + ["measurement_total_s", "compute_total_ms"]
        stats = steps.groupby("paradigm")[keys].agg(["mean", "std"]).reindex(
            [p for p in OVERHEAD_PARADIGMS if p in set(steps["paradigm"])])
        stats.columns = [f"{c}_{s}" for c, s in stats.columns]
        write_csv(stats.reset_index(), os.path.join(out, "overhead_steps_summary.csv"))
        means = steps.groupby("paradigm")[keys].mean()
        n = steps["scenario"].nunique()
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
        _stacked(axes[0], means, COMPUTE_STEPS, "ms", "Decision compute and installation time (ms)",
                 f"GEANT2 live run · mean over {n} scenarios · inference: one batched call for all K candidates")
        _stacked(axes[1], means, MEASUREMENT_STEPS, "s", "Measurement time per decision (s)",
                 f"GEANT2 live run · mean over {n} scenarios · hop-count needs no telemetry")
        fig.subplots_adjust(left=0.17, right=0.99, wspace=0.62, top=0.9, bottom=0.42)
        fig.savefig(os.path.join(out, "overhead_breakdown.png"))
        plt.close(fig)

    if os.path.exists(benchmark_csv):
        bench = pd.read_csv(benchmark_csv)
        shutil.copyfile(benchmark_csv, os.path.join(out, os.path.basename(benchmark_csv)))
        vb.plot_offline_latency(benchmark_csv, os.path.join(out, "offline_compute_per_method.png"), "GEANT2")
        if "stage2_sequential_ms" in bench:
            summary = bench.groupby("k")[["stage2_inference_ms", "stage2_sequential_ms"]].agg(["mean", "std"])
            summary.columns = [f"{c}_{s}" for c, s in summary.columns]
            summary = summary.reset_index()
            write_csv(summary, os.path.join(out, "batching_summary.csv"))
            ks = sorted(bench["k"].unique())
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            width = 0.36
            for j, (column, label, color) in enumerate((("stage2_inference_ms", "batched: one call, all K", RAMP[3]),
                                                        ("stage2_sequential_ms", "one call per candidate", RAMP[0]))):
                values = [summary.loc[summary["k"] == k, f"{column}_mean"].iloc[0] for k in ks]
                xs = np.arange(len(ks)) + (j - 0.5) * width
                ax.bar(xs, values, width * 0.94, color=color, zorder=2, label=label)
                for xi, v in zip(xs, values):
                    ax.annotate(f"{v:.1f}", xy=(xi, v), xytext=(0, 3), textcoords="offset points", ha="center",
                                fontsize=8, color=INK)
            ax.set_xticks(range(len(ks)), [f"K = {k}" for k in ks])
            ax.set_title("Stage 2 inference time (ms)", pad=16)
            ax.text(0, 1.02, f"GEANT2 ensemble · mean of {bench['trial'].nunique()} trials per method and K",
                    transform=ax.transAxes, fontsize=7.6, color=MUTED, va="bottom")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(axis="x", visible=False)
            fig.tight_layout()
            fig.savefig(os.path.join(out, "batched_vs_sequential_inference.png"))
            plt.close(fig)


# ----------------------------------------------------------------------------------------
# 5. Chosen paths
# ----------------------------------------------------------------------------------------
def layout(spec) -> dict:
    if spec.name == "fattree":
        pos = {f"s{i}": (1.0 + 2 * i, 3.0) for i in range(4)}                  # core
        pos.update({f"s{i}": (float(i - 4) + (i - 4) // 2, 2.0) for i in range(4, 12)})   # aggregation
        pos.update({f"s{i}": (float(i - 12) + (i - 12) // 2, 1.0) for i in range(12, 20)})  # edge
        return pos
    return nx.kamada_kawai_layout(spec.graph)


def draw_paths(ax, spec, pos, load: dict, route: list, truth: dict, title: str) -> None:
    graph = spec.graph
    nx.draw_networkx_edges(graph, pos, edgelist=list(zip(route, route[1:])), ax=ax, edge_color=INK, width=7.0)
    for u, v in graph.edges:
        c = class_of(load[link_key(u, v)])
        nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], ax=ax, edge_color=CLASS_COLORS[c], width=2.8 if c else 2.0)
    others = [n for n in graph.nodes if n not in route]
    nx.draw_networkx_nodes(graph, pos, nodelist=others, ax=ax, node_color=SURFACE, edgecolors=MUTED, linewidths=0.9,
                           node_size=170)
    nx.draw_networkx_nodes(graph, pos, nodelist=route, ax=ax, node_color=SURFACE, edgecolors=INK, linewidths=1.8,
                           node_size=200)
    nx.draw_networkx_labels(graph, pos, labels={n: n[1:] for n in graph.nodes}, ax=ax, font_size=6, font_color=INK)
    for host, role in ((spec.sender, "Sender"), (spec.receiver, "Receiver")):
        ax.annotate(role, xy=pos[spec.hosts[host].switch], xytext=(0, -12), textcoords="offset points", ha="center",
                    va="top", fontsize=7.4, color=INK, fontweight="bold")
    ax.set_title(title, fontsize=9.5, pad=14)
    if truth:
        ax.text(0, 1.0, f"{len(route) - 1} hops · bottleneck {truth['class_name']} ({truth['bottleneck_load']:.2f})",
                transform=ax.transAxes, fontsize=7.4, color=MUTED, va="bottom")
    ax.axis("off")
    ax.margins(0.07)


def paths(out: str, runs: dict) -> None:
    overview = []
    for topology, (data, table, meta) in runs.items():
        directory = os.path.join(out, topology)
        os.makedirs(directory, exist_ok=True)
        spec = build_topology(topology)
        pos = layout(spec)
        rows, loads = [], []
        for k, paradigms in data.items():
            run_dir = meta["_run_dir"]
            with open(os.path.join(run_dir, f"scenario_{k}", "scenario.json")) as handle:
                utilization = json.load(handle)["utilization"]
            load = {link_key(u, v): utilization[f"{link_key(u, v)[0]}-{link_key(u, v)[1]}"] for u, v in spec.graph.edges}
            loads += [{"scenario": k, "link": f"{u}-{v}", "utilization": round(x, 4), "class": class_of(x),
                       "class_name": CLASS_NAMES[class_of(x)]} for (u, v), x in sorted(load.items())]
            order = [p for p in vb.PARADIGMS if p in paradigms]
            fig, axes = plt.subplots(1, len(order), figsize=(3.4 * len(order), 3.9), squeeze=False)
            for ax, paradigm in zip(axes[0], order):
                decision = paradigms[paradigm]["decision"]
                route, truth = decision.get("chosen_path") or [], decision.get("chosen_ground_truth") or {}
                draw_paths(ax, spec, pos, load, route, truth, LABELS[paradigm])
                candidates = [c["switches"] for c in decision.get("stage1", {}).get("paths", [])]
                candidate_truth = decision.get("candidates_ground_truth") or []
                predicted = [s["predicted_name"] for s in decision.get("stage2", {}).get("scored", [])]
                rows.append({"scenario": k, "paradigm": paradigm, "chosen_path": "-".join(route),
                             "hops": len(route) - 1, "true_class": truth.get("class"),
                             "true_class_name": truth.get("class_name"), "bottleneck_load": truth.get("bottleneck_load"),
                             "stage1_candidates": " | ".join("-".join(c) for c in candidates),
                             "candidate_true_classes": " | ".join(str((g or {}).get("class_name")) for g in candidate_truth),
                             "candidate_predicted_classes": " | ".join(predicted)})
            handles = [Line2D([0], [0], color=CLASS_COLORS[c], linewidth=3.5, label=CLASS_NAMES[c]) for c in range(4)]
            handles.append(Line2D([0], [0], color=INK, linewidth=6, label="chosen path"))
            fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.01),
                       title="link class (offered background load)", title_fontsize=8)
            fig.suptitle(f"{TOPO_LABEL[topology]} · scenario {k}: path chosen by each method", x=0.01, ha="left",
                         fontsize=11, fontweight="bold")
            fig.tight_layout(rect=(0, 0.1, 1, 0.97))
            fig.savefig(os.path.join(directory, f"scenario_{k}_paths.png"))
            plt.close(fig)
        frame = pd.DataFrame(rows)
        write_csv(frame, os.path.join(directory, "chosen_paths.csv"))
        write_csv(pd.DataFrame(loads), os.path.join(directory, "link_loads.csv"))
        overview.append((topology, frame))

    fig, axes = plt.subplots(1, len(overview), figsize=(6.0 * len(overview), 3.8), squeeze=False)
    for ax, (topology, frame) in zip(axes[0], overview):
        order = [p for p in vb.PARADIGMS if p in set(frame["paradigm"])]
        scenarios = sorted(frame["scenario"].unique())
        for i, paradigm in enumerate(order):
            for j, k in enumerate(scenarios):
                row = frame[(frame["paradigm"] == paradigm) & (frame["scenario"] == k)]
                if row.empty:
                    continue
                cls = row["true_class"].iloc[0]
                color = CLASS_COLORS[int(cls)] if pd.notna(cls) else SURFACE
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor=SURFACE, linewidth=2))
                ax.text(j + 0.5, i + 0.5, f"{int(row['hops'].iloc[0])} hops", ha="center", va="center", fontsize=7.6,
                        color=SURFACE if pd.notna(cls) and cls >= 2 else INK)
        ax.set_xlim(0, len(scenarios))
        ax.set_ylim(len(order), 0)
        ax.set_xticks(np.arange(len(scenarios)) + 0.5, [f"S{k}" for k in scenarios])
        ax.set_yticks(np.arange(len(order)) + 0.5, [LABELS[p] for p in order], fontsize=8)
        ax.tick_params(length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{TOPO_LABEL[topology]}: chosen path per scenario", pad=16)
        ax.text(0, 1.02, "cell colour = true class of the chosen path's bottleneck link", transform=ax.transAxes,
                fontsize=7.6, color=MUTED, va="bottom")
    fig.legend(handles=[Patch(facecolor=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in range(4)], loc="lower center",
               ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=3)
    fig.savefig(os.path.join(out, "chosen_path_overview.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------------------
# Raw data and README
# ----------------------------------------------------------------------------------------
def readme(out: str, runs: dict, notes: list[str]) -> None:
    lines = ["# Final results: GEANT2 vs fat-tree (k=4)", "",
             "Every figure sits next to the CSV it was drawn from. Generated by `routing/final_results.py`.", "",
             "## Runs", "", "| Topology | Run | Scenarios | Seeds | Congested default path | Composite weights (α, β, γ, δ) | K |",
             "|---|---|---|---|---|---|---|"]
    for topology, (data, _, meta) in runs.items():
        a = meta.get("args", {})
        seeds = sorted(int(s.get("seed")) for s in meta.get("scenarios", {}).values() if s.get("seed") is not None)
        lines.append(f"| {TOPO_LABEL[topology]} | `{os.path.basename(meta['_run_dir'])}` | {len(data)} | "
                     f"{seeds[0] if seeds else '?'}–{seeds[-1] if seeds else '?'} | {a.get('congest_default_path')} | "
                     f"{a.get('alpha')}, {a.get('beta')}, {a.get('gamma')}, {a.get('delta')} | {a.get('k_paths')} |")
    lines += ["", "## Mean results per method", ""]
    for topology, (_, table, _) in runs.items():
        lines += [f"### {TOPO_LABEL[topology]}", "",
                  "| Method | Throughput (Mbit/s) | Video loss (%) | RTT (ms) | Jitter (ms) | PSNR (dB) | SSIM | Chosen class |",
                  "|---|---|---|---|---|---|---|---|"]
        means = table.groupby("paradigm")[["mean_throughput_mbps", "packet_loss_pct", "mean_rtt_ms", "mean_jitter_ms",
                                           "psnr_db", "ssim", "chosen_true_class"]].agg(
            lambda s: pd.to_numeric(s, errors="coerce").mean())
        fmt = lambda v, f: "n/a" if pd.isna(v) else f.format(v)  # noqa: E731
        for p in present(table):
            m = means.loc[p]
            lines.append(f"| {LABELS[p]} | {fmt(m.mean_throughput_mbps, '{:.2f}')} | {fmt(m.packet_loss_pct, '{:.1f}')} | "
                         f"{fmt(m.mean_rtt_ms, '{:.0f}')} | {fmt(m.mean_jitter_ms, '{:.1f}')} | {fmt(m.psnr_db, '{:.1f}')} | "
                         f"{fmt(m.ssim, '{:.3f}')} | {fmt(m.chosen_true_class, '{:.2f}')} |")
        lines.append("")
    lines += ["RTT is n/a where every echo was lost. Chosen class: true class of the chosen path's bottleneck "
              "(0 Extremely Low … 3 Extremely High).", "",
              "## Contents", "",
              "| Folder | Figure | Data |", "|---|---|---|",
              "| `0_dataset_profiling/` | `feature_correlation_both.png`, `<topology>/charts/*.png`, "
              "`<topology>/<topology>_profile.html` (ydata-profiling report) | `<topology>/<topology>_profile.json`, "
              "`feature_correlation_both.csv`, `profiling_summary.csv`, `<topology>/alerts.txt` |",
              "| `1_training/` | `ensemble_confusion_matrices.png`, `<topology>_model_comparison.png` | "
              "`model_comparison_both.csv`, `<topology>_confusion_matrix_*.csv` |",
              "| `2_path_selection/` | `throughput.png`, `rtt.png`, `jitter.png`, `psnr.png`, `ssim.png` | "
              "`per_scenario.csv`, `summary_mean_std.csv` |",
              "| `2_path_selection/` | `packet_loss_boxplot.png` | `packet_loss_per_second.csv` (boxes), "
              "`per_scenario.csv` `packet_loss_pct` (diamonds: whole-stream video loss) |",
              "| `3_video_timeseries/<topology>/` | `scenario_<k>.png` | `timeseries.csv` |",
              "| `4_overhead_geant2/` | `overhead_breakdown.png` | `overhead_steps_per_scenario.csv`, "
              "`overhead_steps_summary.csv` |",
              "| `4_overhead_geant2/` | `batched_vs_sequential_inference.png`, `offline_compute_per_method.png` | "
              "`decision_latency_benchmark_geant2.csv`, `batching_summary.csv` |",
              "| `5_paths/` | `chosen_path_overview.png`, `<topology>/scenario_<k>_paths.png` | "
              "`<topology>/chosen_paths.csv`, `<topology>/link_loads.csv` |",
              "| `runs/<topology>_<timestamp>/` | – | the raw runs as written by `run_experiment.py` (decision.json, "
              "timeseries.csv, frames.csv, summary.json per scenario and method) |", "",
              "## Methods", "",
              "| Method | Stage 1 (top-K) | Stage 2 |", "|---|---|---|",
              "| Hop-count | hop count, K = 1 | – |",
              "| Latency | measured LLDP delay, K = 1 | – |",
              "| Bandwidth (widest) | max-bottleneck available bandwidth, K = 1 | – |",
              "| Baseline: hop-count + ML | hop count, K = 5 | probe all 5, soft-voting ensemble, lowest class |",
              "| Ours: combined + ML | composite α·D + β·J + γ/BW + δ·L, K = 5 | probe all 5, soft-voting ensemble, "
              "lowest class (ties: composite cost) |", "",
              "Each topology uses its own ensemble (XGBoost, Random Forest, Gradient Boosting, SVM; soft vote), "
              "trained on that topology's dataset. Stage 2 predicts all K candidates in one batched call.", ""]
    if notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    with open(os.path.join(out, "README.md"), "w") as handle:
        handle.write("\n".join(lines))


def build(out: str, run_dirs: dict, benchmark_dir: str = BENCHMARK_DIR) -> str:
    vb.apply_style()
    os.makedirs(out, exist_ok=True)
    runs, notes = {}, []
    for topology in TOPOLOGIES:
        run_dir = run_dirs.get(topology)
        if not run_dir:
            notes.append(f"{TOPO_LABEL[topology]}: no run found; its live results are missing")
            continue
        rescore_run(run_dir)            # videos whose scoring failed during the run (no PSNR yet)
        data, table, meta = load(run_dir)
        meta["_run_dir"] = run_dir
        runs[topology] = (data, table, meta)
    if not runs:
        raise ValueError(f"No runs found under {RUNS_DIR}; run run_experiment.py first")
    notes += training(os.path.join(out, "1_training"))
    path_selection(os.path.join(out, "2_path_selection"), runs)
    video_timeseries(os.path.join(out, "3_video_timeseries"), runs)
    overhead(os.path.join(out, "4_overhead_geant2"), runs["geant2"][0] if "geant2" in runs else None,
             os.path.join(benchmark_dir, "decision_latency_benchmark_geant2.csv"))
    paths(os.path.join(out, "5_paths"), runs)
    readme(out, runs, notes)
    log.info("Wrote %s", out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geant2-run", default=newest_run("geant2"))
    parser.add_argument("--fattree-run", default=newest_run("fattree"))
    parser.add_argument("--benchmark-dir", default=BENCHMARK_DIR)
    parser.add_argument("--out", default=os.path.join(PROJECT_DIR, "results"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        build(args.out, {"geant2": args.geant2_run, "fattree": args.fattree_run}, args.benchmark_dir)
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
