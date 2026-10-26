"""
make_report_figures.py - the figures of the project report (report/final-report.pdf): the architecture diagram,
the topology figure and the profiling-histogram composite are drawn here; the rest are copied from results/.

    .venv/bin/python report/make_report_figures.py      # -> report/figures/
"""
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
FINAL = os.path.join(PROJECT, "results")
OUT = os.path.join(HERE, "figures")
sys.path.insert(0, os.path.join(PROJECT, "routing"))
from final_results import layout  # noqa: E402
from topology_builder import build_topology  # noqa: E402
from visualize_benchmarks import INK, INK_2, MUTED, SURFACE, apply_style  # noqa: E402

STAGE = {"io": ("#fdf3dc", "#c98500"), "one": ("#e8f1fb", "#2a78d6"), "two": ("#e6f6ef", "#1baf7a"),
         "out": ("#f3f2ee", "#898781")}
HIGHLIGHT = "#eda100"


def box(ax, x, y, w, h, title, body="", edge=INK_2, width=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor="#ffffff", edgecolor=edge, linewidth=width))
    ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=9.0, color=INK, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h - 0.062, body, ha="center", va="top", fontsize=7.6, color=INK_2, linespacing=1.35)


def group(ax, x, y, w, h, title, key):
    face, edge = STAGE[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.015",
                                facecolor=face, edgecolor=edge, linewidth=1.3))
    ax.text(x + 0.012, y + h - 0.013, title, ha="left", va="top", fontsize=9.4, color=INK, fontweight="bold")


def arrow(ax, start, end, color=INK_2, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, color=color, linewidth=1.2,
                                 linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2))


def pipeline(path: str) -> None:
    """Fig. 1: the two-stage router."""
    fig, ax = plt.subplots(figsize=(10, 7.8))
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.005, 1.005)
    ax.axis("off")

    group(ax, 0.01, 0.80, 0.98, 0.195, "Network state: LinkTelemetry (6 samples, 2.5 s apart, from Floodlight)", "io")
    box(ax, 0.03, 0.815, 0.28, 0.13, "Switch port counters",
        "rate per link → utilisation,\navailable bandwidth, loss")
    ax.text(0.33, 0.88, "+", ha="center", va="center", fontsize=14, color=INK_2)
    box(ax, 0.35, 0.815, 0.28, 0.13, "LLDP link latency", "delay = mean of the samples;\njitter = their std. dev.")
    arrow(ax, (0.63, 0.88), (0.69, 0.88))
    box(ax, 0.69, 0.815, 0.28, 0.13, "Annotated topology graph", "GEANT2 or fat-tree links:\ndelay, jitter, avail. BW, loss")

    group(ax, 0.01, 0.53, 0.98, 0.25,
          "Stage 1: multi-metric candidate discovery (top-K loopless paths, K = 5–10, hops ≤ shortest + 4)", "one")
    methods = [("Hop count", "link weight 1\n(baseline\ncandidates)", INK_2, 1.1),
               ("Latency", "link weight =\nmeasured delay", INK_2, 1.1),
               ("Available bandwidth", "K widest paths\n(max-bottleneck\nDijkstra)", INK_2, 1.1),
               ("Composite (default)", "α·D/Dmax + β·J/Jmax\n+ γ/(B/Bmax) + δ·L\nα .3, β .1, γ .4, δ .2",
                HIGHLIGHT, 2.2)]
    for i, (title, body, edge, width) in enumerate(methods):
        box(ax, 0.03 + i * 0.24, 0.575, 0.22, 0.15, title, body, edge=edge, width=width)
    arrow(ax, (0.83, 0.815), (0.83, 0.78))
    ax.text(0.5, 0.543, "every candidate keeps its composite cost, whatever method produced it",
            ha="center", va="bottom", fontsize=7.6, color=MUTED, style="italic")

    group(ax, 0.01, 0.21, 0.98, 0.295, "Stage 2: multi-class ML selection", "two")
    box(ax, 0.03, 0.32, 0.29, 0.14, "Probe each candidate",
        "ICMP/TCP pinned at priority 40000:\n50 pings · 5 s iperf3 TCP · 2 s drain\n→ 11 features")
    box(ax, 0.355, 0.32, 0.29, 0.14, "Soft-voting ensemble",
        "XGBoost · Random Forest ·\nGradient Boosting · SVM (RBF)\none batched call → P(class 0–3)")
    box(ax, 0.68, 0.32, 0.29, 0.14, "Select", "lowest predicted class;\nties → lowest composite cost,\nthen Stage-1 rank")
    arrow(ax, (0.32, 0.39), (0.355, 0.39))
    arrow(ax, (0.645, 0.39), (0.68, 0.39))
    arrow(ax, (0.175, 0.53), (0.175, 0.505))
    face, edge = STAGE["io"]
    ax.add_patch(FancyBboxPatch((0.335, 0.225), 0.33, 0.07, boxstyle="round,pad=0.006,rounding_size=0.01",
                                facecolor=face, edgecolor=edge, linewidth=1.0, linestyle="--"))
    ax.text(0.5, 0.26, "offline, per topology: GEANT2 (500 paths) or fat-tree\n(896 paths) → train_models.py → model bundle",
            ha="center", va="center", fontsize=7.4, color=INK_2)
    arrow(ax, (0.5, 0.295), (0.5, 0.32), dashed=True)

    group(ax, 0.01, 0.01, 0.98, 0.18, "Forwarding and evaluation", "out")
    box(ax, 0.53, 0.025, 0.44, 0.12, "Install route",
        "Floodlight StaticEntryPusher, OpenFlow 1.3:\nall Sender ↔ Receiver IPv4 traffic, both directions")
    box(ax, 0.03, 0.025, 0.44, 0.12, "Multimedia evaluator",
        "H.264 stream Sender → Receiver: per-second RTT, jitter,\nthroughput, loss; per-frame PSNR and SSIM")
    arrow(ax, (0.825, 0.32), (0.825, 0.145))
    arrow(ax, (0.53, 0.085), (0.47, 0.085))
    fig.savefig(path)
    plt.close(fig)


def topologies(path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, name, title in zip(axes, ("geant2", "fattree"),
                               ("GEANT2: 24 switches, 37 links", "Fat-tree (k = 4): 20 switches, 32 links")):
        spec = build_topology(name)
        pos = layout(spec)
        ends = {spec.hosts[spec.sender].switch: "Sender", spec.hosts[spec.receiver].switch: "Receiver"}
        nx.draw_networkx_edges(spec.graph, pos, ax=ax, edge_color="#c3c2b7", width=1.6)
        nx.draw_networkx_nodes(spec.graph, pos, ax=ax, nodelist=[n for n in spec.graph if n not in ends],
                               node_color=SURFACE, edgecolors=MUTED, linewidths=1.0, node_size=260)
        nx.draw_networkx_nodes(spec.graph, pos, ax=ax, nodelist=list(ends), node_color="#eda100", edgecolors=INK,
                               linewidths=1.6, node_size=300)
        nx.draw_networkx_labels(spec.graph, pos, labels={n: n[1:] for n in spec.graph}, ax=ax, font_size=7, font_color=INK)
        for switch, role in ends.items():
            ax.annotate(f"{role} ({switch})", xy=pos[switch], xytext=(0, -14), textcoords="offset points", ha="center",
                        va="top", fontsize=8, color=INK, fontweight="bold")
        if name == "fattree":
            for y, label in ((3, "core"), (2, "aggregation"), (1, "edge")):
                ax.text(-1.6, y, label, ha="right", va="center", fontsize=8, color=INK_2)
        ax.set_title(title, pad=10)
        ax.axis("off")
        ax.margins(0.08)
    fig.tight_layout(w_pad=3)
    fig.savefig(path)
    plt.close(fig)


def profile_histograms(path: str) -> None:
    """Four ydata-profiling histograms per dataset, laid out side by side."""
    variables = [("throughput_mbps", "TCP throughput (Mbit/s)"), ("rtt_avg_ms", "mean RTT (ms)"),
                 ("path_max_link_util_pct", "max link utilisation on path (%)"), ("hop_count", "hop count")]
    fig, axes = plt.subplots(2, len(variables), figsize=(14, 5.6))
    for row, (name, label) in enumerate((("geant2", "GEANT2"), ("fattree", "Fat-tree"))):
        for col, (var, title) in enumerate(variables):
            charts = os.path.join(FINAL, "0_dataset_profiling", name, "charts")
            candidates = [f"{var}_histogram.png", f"{var}_common_values.png"]
            file = next(os.path.join(charts, c) for c in candidates if os.path.exists(os.path.join(charts, c)))
            ax = axes[row, col]
            ax.imshow(mpimg.imread(file))
            ax.axis("off")
            ax.set_title(f"{label}: {title}", fontsize=9, pad=4)
    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    fig.savefig(path, dpi=220)
    plt.close(fig)


COPIES = {
    "correlation.png": "0_dataset_profiling/feature_correlation_both.png",
    "confusion.png": "1_training/ensemble_confusion_matrices.png",
    "throughput.png": "2_path_selection/throughput.png", "rtt.png": "2_path_selection/rtt.png",
    "jitter.png": "2_path_selection/jitter.png", "psnr.png": "2_path_selection/psnr.png",
    "ssim.png": "2_path_selection/ssim.png", "loss_box.png": "2_path_selection/packet_loss_boxplot.png",
    "ts_geant2_s0.png": "3_video_timeseries/geant2/scenario_0.png",
    "ts_fattree_s3.png": "3_video_timeseries/fattree/scenario_3.png",
    "overhead.png": "4_overhead_geant2/overhead_breakdown.png",
    "batching.png": "4_overhead_geant2/batched_vs_sequential_inference.png",
    "paths_geant2_s0.png": "5_paths/geant2/scenario_0_paths.png",
    "paths_overview.png": "5_paths/chosen_path_overview.png",
}


def main() -> None:
    apply_style()
    os.makedirs(OUT, exist_ok=True)
    pipeline(os.path.join(OUT, "pipeline.png"))
    topologies(os.path.join(OUT, "topologies.png"))
    profile_histograms(os.path.join(OUT, "profile_histograms.png"))
    for name, source in COPIES.items():
        shutil.copyfile(os.path.join(FINAL, source), os.path.join(OUT, name))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
