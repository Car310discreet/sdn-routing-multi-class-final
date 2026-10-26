"""
run_experiment.py - AIRSDN baseline: compare three routing approaches on NSFNET.

  hop_count  Floodlight's default routing (reactive forwarding, hop-count metric)
  qos_aware  Floodlight's QoS-aware routing (reactive forwarding, latency metric)
  airsdn     AIRSDN: 5 hop-count candidates (Yen) -> 19 features -> Logistic Regression ->
             static route on the path with the lowest predicted traffic level

All three run on one network with the paper's background traffic. For each approach the route
in use is monitored (60 s RTT, 60 s throughput) and the video is streamed and scored (PSNR, SSIM).

    ./start_floodlight.sh                                   # separate terminal
    sudo .venv/bin/python airsdn_reproduction/run_experiment.py
    .venv/bin/python airsdn_reproduction/visualize_benchmarks.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multimedia_evaluator import DEFAULT_VIDEO, Evaluator  # noqa: E402
from path_discovery import K_PATHS, discover_paths  # noqa: E402
from telemetry_and_model import (ControllerError, FloodlightClient, FlowInstaller, PathProber,  # noqa: E402
                                 TrafficClassifier, routing_algorithm, wait_for_switches)
from topology_builder import CLIENT, SERVER, BackgroundTraffic, build_nsfnet, start_network, stop_network  # noqa: E402

log = logging.getLogger("run_experiment")
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(HERE)
APPROACHES = ["hop_count", "qos_aware", "airsdn"]
FLOODLIGHT_METRIC = {"hop_count": "hopcount", "qos_aware": "latency", "airsdn": "hopcount"}


def write_json(path: str, data) -> None:
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, default=str)


def chown_to_invoking_user(path: str) -> None:
    """Results are written as root under sudo; hand them back to the user who ran sudo."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid is None or gid is None:
        return
    for root, dirs, files in os.walk(path):
        for name in [root] + [os.path.join(root, n) for n in dirs + files]:
            try:
                os.chown(name, int(uid), int(gid))
            except OSError:
                pass


def run(args) -> str:
    if os.geteuid() != 0:
        raise SystemExit("Mininet needs root: run with sudo")
    api = FloodlightClient(args.controller_url)
    if not api.health():
        raise SystemExit(f"Floodlight is not reachable at {args.controller_url} (start ./start_floodlight.sh)")
    classifier = TrafficClassifier(args.model)
    spec = build_nsfnet()
    run_dir = os.path.join(args.out, f"nsfnet_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)
    metadata = {"topology": "nsfnet", "client": CLIENT, "server": SERVER, "model": args.model, "approaches": {}}

    net = background = None
    installed: list[str] = []
    installer = FlowInstaller(api, spec)
    try:
        net = start_network(spec)
        wait_for_switches(api, set(spec.switch_dpids.values()))
        api.clear_flows()            # static entries of earlier runs reappear on reused DPIDs
        api.set_routing_metric("hopcount")
        log.info("Testing connectivity (pingAll) so Floodlight learns every host")
        net.pingAll()
        background = BackgroundTraffic(net, spec, installer)
        background.start()
        time.sleep(13)               # as in the paper's original experiment scripts, let the background traffic ramp up

        evaluator = Evaluator(net, spec, SERVER, CLIENT, args.video)
        for approach in args.approaches:
            log.info("=== %s ===", approach)
            installer.remove(installed)
            installed = []
            api.set_routing_metric(FLOODLIGHT_METRIC[approach])
            time.sleep(6)            # reactive entries of the previous approach idle out (5 s timeout)
            out_dir = os.path.join(run_dir, approach)
            os.makedirs(out_dir, exist_ok=True)
            decision = {"approach": approach, "floodlight_metric": FLOODLIGHT_METRIC[approach]}
            if approach == "airsdn":
                stage1 = discover_paths(spec.graph, spec.hosts[CLIENT].switch, spec.hosts[SERVER].switch, K_PATHS)
                prober = PathProber(net, spec, probe_seconds=args.probe_seconds)
                selection, installed = routing_algorithm(stage1.paths, CLIENT, SERVER, installer, prober, classifier)
                decision.update(stage1=stage1.to_dict(), stage2=selection.to_dict(), chosen_path=selection.chosen)
                log.info("AIRSDN chose path %d: %s", selection.chosen_index, "-".join(selection.chosen))
            write_json(os.path.join(out_dir, "decision.json"), decision)
            summary = evaluator.run(approach, out_dir)
            metadata["approaches"][approach] = {"summary": summary.__dict__,
                                                "chosen_path": decision.get("chosen_path")}
            write_json(os.path.join(run_dir, "run.json"), metadata)
    finally:
        if background is not None:
            background.stop()
        installer.remove(installed)
        if net is not None:
            try:
                api.clear_flows()    # while switches are connected, so nothing leaks into the next run
            except ControllerError as exc:
                log.warning("Could not clear static entries: %s", exc)
        stop_network(net)
        write_json(os.path.join(run_dir, "run.json"), metadata)
        chown_to_invoking_user(args.out)
    log.info("Results in %s", run_dir)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approaches", nargs="+", default=APPROACHES, choices=APPROACHES)
    parser.add_argument("--model", default=os.path.join(HERE, "models", "logistic_regression_model.pkl"))
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--probe-seconds", type=int, default=30, help="iperf3 probe length (training data: 30)")
    parser.add_argument("--controller-url", default="http://127.0.0.1:8080")
    parser.add_argument("--out", default=os.path.join(HERE, "results", "runs"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("mininet").propagate = False   # Mininet prints its own output; avoid duplicates
    try:
        run(args)
    except (ControllerError, FileNotFoundError, ValueError, TypeError) as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
