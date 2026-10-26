"""
run_experiment.py - live evaluation of the five routing methods on GEANT2 or the k=4 fat-tree (needs root).

Methods (report Section VII-A), all on the same network, scenario and video:
  hop_count   Stage 1 hop count, K = 1                                     shortest path
  latency     Stage 1 lowest measured LLDP delay, K = 1                    latency routing
  bandwidth   Stage 1 widest path on measured available bandwidth, K = 1   bandwidth routing
  baseline    Stage 1 hop count, K = 5, Stage 2 ensemble                   AIRSDN-style candidates (hop + ML)
  proposed    Stage 1 composite cost, K = 5, Stage 2 ensemble              the proposed two-stage router
Every method installs its route as static OpenFlow entries for all Sender <-> Receiver traffic. Stage 2
uses the topology's own model bundle (models/<topology>_bundle.joblib). Hop-count routing needs no link
telemetry, so it does not collect any.

Per scenario (seeds --seed, --seed + 1, ...; four-class link loads, and with --congest-default-path every
link of the hop-count shortest path forced to Extremely High), for each method:
  1. link telemetry snapshot (6 samples, 2.5 s apart): delay, jitter, available bandwidth, loss per link
  2. Stage 1 on the annotated graph -> K candidates
  3. Stage 2 (baseline, proposed): probe each candidate (11 features), classify all K in one batched
     ensemble call, pick the lowest class (ties: composite cost, then Stage-1 rank)
  4. install the route, stream the video, log per-second QoS and per-frame PSNR / SSIM
Every step is timed (decision.json "timing"): telemetry_s, stage1_ms, probe_install_ms, ping_s, iperf_s,
drain_s, probe_remove_ms, assembly_ms, inference_ms, select_ms, route_install_ms.
Results: results/runs/<topology>_<timestamp>/scenario_<k>/<method>/
    decision.json  candidates (with ground-truth class), Stage-2 scores, chosen path, timings
    timeseries.csv, frames.csv, summary.json
and scenario_<k>/scenario.json, run.json.
--benchmark-only times Stage 1 and Stage 2 inference offline (batched vs one call per candidate) and writes
results/benchmark/decision_latency_benchmark_<topology>.csv.

    ./start_floodlight.sh                                                         # separate terminal
    sudo .venv/bin/python routing/run_experiment.py --topology geant2 --scenarios 5 --congest-default-path
    .venv/bin/python routing/run_experiment.py --benchmark-only --topology geant2   # no root

An interrupted run continues in its directory, redoing scenario k in full:
    sudo .venv/bin/python routing/run_experiment.py ... --resume results/runs/<run> --first-scenario k
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topology_builder import (CLASS_NAMES, TOPOLOGIES, TrafficGenerator, build_topology,  # noqa: E402
                              chown_to_invoking_user, sample_scenario, start_topology, stop_network)
from path_discovery import DEFAULT_HOP_SLACK, METHODS, discover_paths, synthetic_link_metrics  # noqa: E402
from telemetry_and_model import (MODEL_PATHS, ROUTE_PRIORITY, ControllerError, FloodlightClient,  # noqa: E402
                                 FlowInstaller, LinkTelemetry, MultiClassModel, MultiClassPathSelector,
                                 PathTelemetry)

log = logging.getLogger("run_experiment")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARADIGMS = ["hop_count", "latency", "bandwidth", "baseline", "proposed"]


def paradigm_plan(paradigm: str, args) -> tuple[str, int, bool]:
    """(Stage-1 method, K, uses Stage 2)"""
    return {"hop_count": ("hop", 1, False), "latency": ("latency", 1, False), "bandwidth": ("bandwidth", 1, False),
            "baseline": ("hop", args.k_paths, True), "proposed": (args.stage1_method, args.k_paths, True)}[paradigm]


def write_json(path: str, data) -> None:
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, default=str)


def ground_truth(scenario, path) -> dict | None:
    if not path:
        return None
    cls = scenario.path_class(path)
    return {"bottleneck_load": round(scenario.path_utilization(path), 4), "class": cls, "class_name": CLASS_NAMES[cls]}


def decide(paradigm, args, spec, link_telemetry, selector, scenario) -> dict:
    method, k, use_ml = paradigm_plan(paradigm, args)
    weights = {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma, "delta": args.delta}
    snapshot, telemetry_s, graph = None, 0.0, spec.graph
    if method != "hop" or use_ml:            # hop count needs no telemetry; Stage 2 uses its utilisation
        started = time.monotonic()
        snapshot = link_telemetry.collect()
        telemetry_s = time.monotonic() - started
        graph = snapshot.annotated_graph(spec.graph)
    src, dst = spec.hosts[spec.sender].switch, spec.hosts[spec.receiver].switch
    stage1 = discover_paths(graph, src, dst, method, k, hop_slack=args.hop_slack, weights=weights)
    decision = {"paradigm": paradigm, "stage1": stage1.to_dict(),
                "candidates_ground_truth": [ground_truth(scenario, p.switches) for p in stage1.paths],
                "timing": {"telemetry_s": round(telemetry_s, 2), "stage1_ms": stage1.compute_ms,
                           "stage2_inference_ms": 0.0, "stage2_probe_s": 0.0}}
    if use_ml:
        stage2 = selector.select(stage1.paths, spec.sender, spec.receiver, snapshot)
        decision["stage2"] = stage2.to_dict()
        decision["chosen_path"] = stage2.chosen
        decision["timing"].update(stage2.timing)
        decision["timing"].update(stage2_inference_ms=stage2.inference_ms, stage2_probe_s=stage2.probe_s)
    else:
        decision["chosen_path"] = stage1.paths[0].switches
    decision["chosen_ground_truth"] = ground_truth(scenario, decision["chosen_path"])
    return decision


def run(args) -> str:
    from multimedia_evaluator import VideoEvaluator
    if os.geteuid() != 0:
        raise SystemExit("Mininet needs root: run with sudo")
    api = FloodlightClient(args.controller_url)
    if not api.health():
        raise SystemExit(f"Floodlight is not reachable at {args.controller_url} (start ./start_floodlight.sh)")
    model = MultiClassModel.load(args.model_path, args.model) if any(
        paradigm_plan(p, args)[2] for p in args.paradigms) else None
    spec = build_topology(args.topology)
    run_dir = args.resume or os.path.join(args.out, f"{args.topology}_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)
    metadata = {"topology": spec.name, "sender": spec.sender, "receiver": spec.receiver, "args": vars(args),
                "model": getattr(model, "name", None), "model_path": args.model_path, "scenarios": {}}
    if args.resume and os.path.exists(os.path.join(run_dir, "run.json")):
        with open(os.path.join(run_dir, "run.json")) as handle:
            metadata["scenarios"] = {int(k): v for k, v in json.load(handle).get("scenarios", {}).items()
                                     if int(k) < args.first_scenario}

    net = generator = None
    installed: list[str] = []
    installer = FlowInstaller(api, spec)
    try:
        net = start_topology(spec, api)
        generator = TrafficGenerator(net, spec, installer)
        generator.start()
        link_telemetry = LinkTelemetry(api, spec, args.telemetry_samples, args.telemetry_interval)
        selector = MultiClassPathSelector(installer, PathTelemetry(net, spec), model) if model else None
        evaluator = VideoEvaluator(net, spec, spec.sender, spec.receiver, args.video)
        default_path = discover_paths(spec.graph, spec.hosts[spec.sender].switch, spec.hosts[spec.receiver].switch,
                                      "hop", 1).paths[0].switches
        for k in range(args.first_scenario, args.scenarios):
            scenario = sample_scenario(spec.graph, args.seed + k, spec.path_hops)
            if args.congest_default_path:
                scenario.congest(default_path)
            generator.apply(scenario)
            time.sleep(args.warmup)
            scenario_dir = os.path.join(run_dir, f"scenario_{k}")
            os.makedirs(scenario_dir, exist_ok=True)
            write_json(os.path.join(scenario_dir, "scenario.json"), scenario.to_dict())
            metadata["scenarios"][k] = {"seed": scenario.seed, "paradigms": {}}
            for paradigm in args.paradigms:
                log.info("=== scenario %d: %s ===", k, paradigm)
                installer.remove(installed)
                installed = []
                time.sleep(2)
                out_dir = os.path.join(scenario_dir, paradigm)
                os.makedirs(out_dir, exist_ok=True)
                decision = decide(paradigm, args, spec, link_telemetry, selector, scenario)
                install_start = time.perf_counter()
                installed = installer.install_path(spec.hosts[spec.sender].ip, spec.hosts[spec.receiver].ip,
                                                   decision["chosen_path"], prefix="route", priority=ROUTE_PRIORITY)
                decision["timing"]["route_install_ms"] = round((time.perf_counter() - install_start) * 1000, 3)
                log.info("%s routes over %s (ground truth: %s; Stage 1 %.2f ms, Stage 2 inference %.2f ms)",
                         paradigm, "-".join(decision["chosen_path"]), decision["chosen_ground_truth"]["class_name"],
                         decision["timing"]["stage1_ms"], decision["timing"]["stage2_inference_ms"])
                write_json(os.path.join(out_dir, "decision.json"), decision)
                summary = evaluator.run(paradigm, out_dir)
                metadata["scenarios"][k]["paradigms"][paradigm] = {
                    "summary": asdict(summary), "chosen_path": decision["chosen_path"],
                    "chosen_ground_truth": decision["chosen_ground_truth"], "timing": decision["timing"]}
                write_json(os.path.join(run_dir, "run.json"), metadata)
    finally:
        installer.remove(installed)
        if generator is not None:
            generator.stop()
        if net is not None:
            try:
                api.clear_flows()
            except ControllerError as exc:
                log.warning("Could not clear static entries: %s", exc)
        stop_network(net)
        write_json(os.path.join(run_dir, "run.json"), metadata)
        chown_to_invoking_user(args.out)
    log.info("Results in %s", run_dir)
    return run_dir


def benchmark(args) -> str:
    """Offline Stage-1 + Stage-2-inference timing with synthetic telemetry (no root, no controller).

    Stage 2 is timed twice on the same feature rows: one batched predict_proba call for all candidates
    (what the selector does) and one call per candidate, which shows what batching saves."""
    import pandas as pd
    model = MultiClassModel.load(args.model_path, args.model)
    data = pd.read_csv(os.path.join(PROJECT_DIR, "data", f"{args.topology}_dataset.csv"))
    spec = build_topology(args.topology)
    src, dst = spec.hosts[spec.sender].switch, spec.hosts[spec.receiver].switch
    rng = random.Random(args.seed)
    rows = []
    for trial in range(args.trials):
        graph = synthetic_link_metrics(spec.graph, seed=args.seed + trial)
        for k in sorted({5, 10, args.k_paths}):
            for method in METHODS:
                result = discover_paths(graph, src, dst, method, k, hop_slack=args.hop_slack)
                rows_k = [data.iloc[rng.randrange(len(data))][model.features].astype(float).to_dict()
                          for _ in result.paths]
                inference = model.predict_batch(rows_k)[2]
                assembly = model.last_assembly_ms
                sequential = sum(model.predict(row)[2] for row in rows_k)
                rows.append({"topology": args.topology, "trial": trial, "method": method, "k": k,
                             "candidates": len(result.paths), "stage1_ms": result.compute_ms,
                             "assembly_ms": round(assembly, 4), "stage2_inference_ms": round(inference, 4),
                             "stage2_sequential_ms": round(sequential, 4),
                             "total_ms": round(result.compute_ms + assembly + inference, 4)})
    os.makedirs(args.benchmark_dir, exist_ok=True)
    path = os.path.join(args.benchmark_dir, f"decision_latency_benchmark_{args.topology}.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = pd.DataFrame(rows).groupby(["method", "k"])[["stage1_ms", "assembly_ms", "stage2_inference_ms",
                                                         "stage2_sequential_ms", "total_ms"]].mean()
    print(table.round(3).to_string())
    log.info("Benchmark (%d trials, model %s) in %s", args.trials, model.name, path)
    chown_to_invoking_user(args.benchmark_dir)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topology", default="geant2", choices=sorted(TOPOLOGIES))
    parser.add_argument("--paradigms", nargs="+", default=PARADIGMS, choices=PARADIGMS)
    parser.add_argument("--stage1-method", default="composite", choices=METHODS,
                        help="Stage-1 method of the combined paradigm")
    parser.add_argument("--k-paths", type=int, default=5, help="candidates for the ML paradigms (5-10)")
    parser.add_argument("--hop-slack", type=int, default=DEFAULT_HOP_SLACK)
    parser.add_argument("--alpha", type=float, default=0.3, help="composite weight: delay")
    parser.add_argument("--beta", type=float, default=0.1, help="composite weight: jitter")
    parser.add_argument("--gamma", type=float, default=0.4, help="composite weight: 1/available bandwidth")
    parser.add_argument("--delta", type=float, default=0.2, help="composite weight: loss")
    parser.add_argument("--model", default="ensemble",
                        choices=["ensemble", "XGBoost", "Random Forest", "Gradient Boosting", "SVM"])
    parser.add_argument("--model-path", help="default: models/<topology>_bundle.joblib")
    parser.add_argument("--scenarios", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000, help="seed of the first scenario")
    parser.add_argument("--congest-default-path", action="store_true",
                        help="force every link of the hop-count shortest path to Extremely High")
    parser.add_argument("--warmup", type=float, default=10.0)
    parser.add_argument("--telemetry-samples", type=int, default=6)
    parser.add_argument("--telemetry-interval", type=float, default=2.5)
    parser.add_argument("--video", default=os.path.join(PROJECT_DIR, "assets", "surreal.ts"))
    parser.add_argument("--controller-url", default="http://127.0.0.1:8080")
    parser.add_argument("--out", default=os.path.join(PROJECT_DIR, "results", "runs"))
    parser.add_argument("--benchmark-dir", default=os.path.join(PROJECT_DIR, "results", "benchmark"))
    parser.add_argument("--resume", metavar="RUN_DIR")
    parser.add_argument("--first-scenario", type=int, default=0)
    parser.add_argument("--benchmark-only", action="store_true", help="offline compute timing (no root)")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    args.model_path = args.model_path or MODEL_PATHS[args.topology]
    if not 1 <= args.k_paths <= 10:
        parser.error("--k-paths must be between 1 and 10")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("mininet").propagate = False
    try:
        benchmark(args) if args.benchmark_only else run(args)
    except (ControllerError, FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
