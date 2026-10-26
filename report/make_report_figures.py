"""
make_report_figures.py - figures for the IEEE report that final_results/ does not contain, plus copies of the
final_results/ figures the report uses (so report_ieee/ compiles on its own).

    .venv/bin/python report_ieee/make_report_figures.py      # -> report_ieee/figures/
"""
import inspect
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
FINAL = os.path.join(PROJECT, "final_results")
OUT = os.path.join(HERE, "figures")
sys.path[:0] = [os.path.join(PROJECT, "advanced"), os.path.join(PROJECT, "advanced", "report")]
import make_figures  # noqa: E402
from final_results import layout  # noqa: E402
from topology_builder import build_topology  # noqa: E402
from visualize_benchmarks import INK, INK_2, MUTED, SURFACE, apply_style  # noqa: E402


def pipeline(path: str) -> None:
    """The branch's architecture diagram with topology-neutral labels (GEANT2 and the fat-tree)."""
    source = inspect.getsource(make_figures.pipeline)
    for old, new in (('"Annotated GEANT2 graph", "24 switches, 37 links:\\ndelay, jitter, avail. BW, loss"',
                      '"Annotated topology graph", "GEANT2 or fat-tree links:\\ndelay, jitter, avail. BW, loss"'),
                     ('"link weight 1\\n(GEANT2-branch\\ncandidates)"', '"link weight 1\\n(baseline\\ncandidates)"'),
                     ('"offline: GEANT2 dataset (500 paths, 4 classes)\\n→ train_models.py → model bundle"',
                      '"offline, per topology: GEANT2 (500 paths) or fat-tree\\n(896 paths) → train_models.py → model bundle"'),
                     ('"H.264 stream h15 → h23: per-second RTT, jitter,', '"H.264 stream Sender → Receiver: per-second RTT, jitter,')):
        assert old in source, old
        source = source.replace(old, new)
    namespace = dict(vars(make_figures))
    exec(source, namespace)
    namespace["pipeline"](path)


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
    "paths_fattree_s1.png": "5_paths/fattree/scenario_1_paths.png",
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
