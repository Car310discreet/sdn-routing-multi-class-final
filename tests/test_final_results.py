"""Offline test of final_results.py: builds the results folder from synthetic GEANT2 and fat-tree runs in tmp_path."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "routing"))
import final_results  # noqa: E402
from path_discovery import METHODS, discover_paths, synthetic_link_metrics  # noqa: E402
from topology_builder import CLASS_NAMES, build_topology, sample_scenario  # noqa: E402

PLAN = {"hop_count": ("hop", 1, False), "latency": ("latency", 1, False), "bandwidth": ("bandwidth", 1, False),
        "baseline": ("hop", 5, True), "proposed": ("composite", 5, True)}


def synthetic_run(root, topology: str, scenarios: int = 2) -> str:
    rng = np.random.default_rng(0)
    spec = build_topology(topology)
    run = os.path.join(root, f"{topology}_20990101_000000")
    src, dst = spec.hosts[spec.sender].switch, spec.hosts[spec.receiver].switch
    meta = {"args": {"congest_default_path": True, "alpha": 0.3, "beta": 0.1, "gamma": 0.4, "delta": 0.2,
                     "k_paths": 5}, "scenarios": {}}
    for k in range(scenarios):
        scenario = sample_scenario(spec.graph, 1000 + k, spec.path_hops)
        directory = os.path.join(run, f"scenario_{k}")
        os.makedirs(directory)
        json.dump(scenario.to_dict(), open(os.path.join(directory, "scenario.json"), "w"))
        meta["scenarios"][str(k)] = {"seed": 1000 + k}
        graph = synthetic_link_metrics(spec.graph, k)
        for paradigm, (method, K, ml) in PLAN.items():
            stage1 = discover_paths(graph, src, dst, method, K)
            chosen = (min(stage1.paths, key=lambda c: scenario.path_class(c.switches)) if ml else stage1.paths[0]).switches
            cls = scenario.path_class(chosen)
            out = os.path.join(directory, paradigm)
            os.makedirs(out)
            timing = {"telemetry_s": 0.0 if paradigm == "hop_count" else 15.0, "stage1_ms": stage1.compute_ms,
                      "route_install_ms": 40.0, "stage2_inference_ms": 30.0 * ml, "stage2_probe_s": 90.0 * ml}
            if ml:
                timing.update(probe_install_ms=900.0, probe_remove_ms=300.0, ping_s=30.0, iperf_s=35.0, drain_s=10.0,
                              assembly_ms=0.02, inference_ms=30.0, select_ms=0.01)
            json.dump({"chosen_path": chosen, "timing": timing, "stage1": stage1.to_dict(),
                       "chosen_ground_truth": {"class": cls, "class_name": CLASS_NAMES[cls],
                                               "bottleneck_load": scenario.path_utilization(chosen)}},
                      open(os.path.join(out, "decision.json"), "w"))
            json.dump({"psnr_db": 36 - 7 * cls, "ssim": 1 - 0.1 * cls, "mean_rtt_ms": None if cls == 3 else 160.0,
                       "mean_jitter_ms": 13.0, "mean_throughput_mbps": 4.6 - cls, "packet_loss_pct": [0, 1, 7, 99][cls],
                       "frames_decoded": 4000}, open(os.path.join(out, "summary.json"), "w"))
            pd.DataFrame({"t_s": range(1, 61), "rtt_ms": rng.normal(160, 20, 60), "jitter_ms": rng.normal(13, 2, 60),
                          "throughput_mbps": rng.normal(4.6 - cls, 0.2, 60),
                          "packet_loss_pct": np.clip(rng.normal([0, 1, 7, 99][cls], 2, 60), 0, 100)}).to_csv(
                os.path.join(out, "timeseries.csv"), index=False)
    json.dump(meta, open(os.path.join(run, "run.json"), "w"))
    return run


def test_final_results_writes_every_figure_with_its_data(tmp_path):
    runs = {t: synthetic_run(tmp_path / "runs", t) for t in ("geant2", "fattree")}
    pd.DataFrame([{"topology": "geant2", "trial": 0, "method": m, "k": k, "stage1_ms": 2.0, "assembly_ms": 0.01,
                   "stage2_inference_ms": 15.0, "stage2_sequential_ms": 70.0 * k / 5}
                  for m in METHODS for k in (5, 10)]).to_csv(tmp_path / "decision_latency_benchmark_geant2.csv",
                                                             index=False)
    out = tmp_path / "results"
    final_results.build(str(out), runs, str(tmp_path))
    expected = ["README.md", "2_path_selection/throughput.png", "2_path_selection/rtt.png", "2_path_selection/jitter.png",
                "2_path_selection/psnr.png", "2_path_selection/ssim.png", "2_path_selection/packet_loss_boxplot.png",
                "2_path_selection/per_scenario.csv", "2_path_selection/packet_loss_per_second.csv",
                "3_video_timeseries/geant2/scenario_1.png", "3_video_timeseries/fattree/timeseries.csv",
                "4_overhead_geant2/overhead_breakdown.png", "4_overhead_geant2/overhead_steps_summary.csv",
                "4_overhead_geant2/batched_vs_sequential_inference.png", "5_paths/chosen_path_overview.png",
                "5_paths/fattree/scenario_0_paths.png", "5_paths/geant2/chosen_paths.csv"]
    for name in expected:
        assert (out / name).exists(), name
    steps = pd.read_csv(out / "4_overhead_geant2" / "overhead_steps_summary.csv").set_index("paradigm")
    assert list(steps.index) == ["hop_count", "baseline", "proposed"]
    assert steps.loc["hop_count", "telemetry_s_mean"] == 0 and steps.loc["proposed", "probe_rules_ms_mean"] == 1200
    per_scenario = pd.read_csv(out / "2_path_selection" / "per_scenario.csv")
    assert set(per_scenario["topology"]) == {"geant2", "fattree"} and per_scenario["paradigm"].nunique() == 5
