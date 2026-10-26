"""
collect_dataset.py - build a topology's four-class path dataset on the live emulated network (needs root and a
running Floodlight). Report Section VI, "Datasets".

Each epoch:
  1. the traffic generator applies a new four-class load scenario and the network warms up (--warmup s). With
     probability --targeted (default 0.7) the scenario targets the class furthest behind its quota, so all
     four classes fill at a similar pace; otherwise it is an untargeted scenario;
  2. one link-telemetry snapshot (port counters 3 s apart) gives the two passive features;
  3. the paths are probed one at a time, their ICMP/TCP pinned with static entries: the 5 hop-count
     candidates Sender -> Receiver, plus one random hop-count candidate of each of --extra-pairs random host
     pairs (on the fat-tree, hosts on edge switches only);
  4. each path is labelled with the class of its bottleneck offered load - the generator's ground truth,
     never a measured value.
Rows are appended as they are measured, so a run can be stopped (Ctrl+C) and resumed with the same command.
A path below 80 % load that lost every ping is a measurement failure, not congestion, and is not written.
Each class is capped at --samples / 4 rows, so the dataset ends up balanced. --trim then cuts every class
to the smallest one by seeded random sampling (the fat-tree dataset: 912 rows -> 896).

The report's datasets (data/geant2_dataset.csv, data/fattree_dataset.csv) are never overwritten: new data
goes to data/collected/<topology>_dataset.csv unless --out says otherwise.

    ./start_floodlight.sh                                                          # separate terminal
    sudo .venv/bin/python routing/collect_dataset.py --topology fattree --samples 1000
    .venv/bin/python routing/collect_dataset.py --trim data/collected/fattree_dataset.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
import sys
import time
from collections import Counter

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topology_builder import (CLASS_BANDS, CLASS_NAMES, FATTREE_EDGE_SWITCHES, TOPOLOGIES, TrafficGenerator,  # noqa: E402
                              build_topology, chown_to_invoking_user, sample_scenario, start_topology, stop_network)
from path_discovery import discover_paths  # noqa: E402
from telemetry_and_model import (FEATURE_NAMES, PROBE_PRIORITY, ControllerError, FloodlightClient,  # noqa: E402
                                 FlowInstaller, LinkTelemetry, PathTelemetry)

log = logging.getLogger("collect_dataset")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_COLUMNS = ["epoch", "time", "src", "dst", "path", "bottleneck_load"]
LABEL_COLUMNS = ["traffic_class", "traffic_class_name"]
PROBE_SWITCHES = {"geant2": None, "fattree": FATTREE_EDGE_SWITCHES}      # None: every switch


def existing_rows(path: str) -> tuple[Counter, int]:
    """Class counts and next epoch number of an existing dataset (for resuming)."""
    counts, epoch = Counter(), 0
    if os.path.exists(path):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                counts[int(row["traffic_class"])] += 1
                epoch = max(epoch, int(row["epoch"]) + 1)
    return counts, epoch


def hop_candidates(graph, a: str, b: str, k: int = 5) -> list[list[str]]:
    return [p.switches for p in discover_paths(graph, a, b, "hop", k).paths]


def epoch_paths(spec, rng: random.Random, candidates: list[list[str]], extra_pairs: int) -> list[tuple]:
    """(src host, dst host, switch path) to probe this epoch."""
    probes = [(spec.sender, spec.receiver, path) for path in candidates]
    switches = sorted(PROBE_SWITCHES[spec.name] or spec.graph.nodes, key=lambda s: int(s[1:]))
    while len(probes) < len(candidates) + extra_pairs:
        a, b = rng.sample(switches, 2)
        if nx.shortest_path_length(spec.graph, a, b) < 2:
            continue                       # adjacent switches: a single link, already pinned by the generator
        probes.append((spec.hosts_on(a)[0], spec.hosts_on(b)[0], rng.choice(hop_candidates(spec.graph, a, b))))
    return probes


def measure_epoch(spec, rng, candidates, args, scenario, counts, quota, installer, telemetry, snapshot) -> list:
    """Probe this epoch's paths; returns (src, dst, path, features, label) rows that passed the checks."""
    rows, pending = [], Counter()
    for index, (src, dst, path) in enumerate(epoch_paths(spec, rng, candidates, args.extra_pairs)):
        label = scenario.path_class(path)
        if counts[label] + pending[label] >= quota:
            continue
        names = installer.install_path(spec.hosts[src].ip, spec.hosts[dst].ip, path, prefix=f"dsprobe{index}",
                                       priority=PROBE_PRIORITY, ip_protos=[1, 6])
        try:
            features = telemetry.measure(src, dst, path, snapshot)
        finally:
            installer.remove(names)
        if features["packet_loss_pct"] >= 100.0 and scenario.path_utilization(path) < CLASS_BANDS[-1][0]:
            log.warning("Discarded %s: every ping lost on a path below %.0f %% load (measurement failure)",
                        "-".join(path), 100 * CLASS_BANDS[-1][0])
            continue
        rows.append((src, dst, path, features, label))
        pending[label] += 1
    return rows


def collect(args) -> str:
    if os.geteuid() != 0:
        raise SystemExit("Mininet needs root: run with sudo")
    api = FloodlightClient(args.controller_url)
    if not api.health():
        raise SystemExit(f"Floodlight is not reachable at {args.controller_url} (start ./start_floodlight.sh)")
    spec = build_topology(args.topology)
    candidates = hop_candidates(spec.graph, spec.hosts[spec.sender].switch, spec.hosts[spec.receiver].switch)
    quota = math.ceil(args.samples / len(CLASS_NAMES))
    counts, first_epoch = existing_rows(args.out)
    log.info("%s: target %d rows, %d per class; already have %s", spec.name, args.samples, quota,
             dict(sorted(counts.items())))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new_file = not os.path.exists(args.out)
    rng = random.Random(args.seed + first_epoch)

    net = generator = None
    installer = FlowInstaller(api, spec)
    try:
        net = start_topology(spec, api)
        generator = TrafficGenerator(net, spec, installer)
        generator.start()
        link_telemetry = LinkTelemetry(api, spec, samples=2, interval_s=3.0)   # one 3 s counter window
        telemetry = PathTelemetry(net, spec, probe_seconds=args.probe_seconds)
        with open(args.out, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=META_COLUMNS + FEATURE_NAMES + LABEL_COLUMNS)
            if new_file:
                writer.writeheader()
            epoch, failures = first_epoch, 0
            while not all(counts[c] >= quota for c in range(len(CLASS_NAMES))):
                if args.max_epochs is not None and epoch >= first_epoch + args.max_epochs:
                    break
                target = None
                if rng.random() < args.targeted:
                    target = min(range(len(CLASS_NAMES)), key=lambda c: (counts[c] / quota, c))
                scenario = sample_scenario(spec.graph, rng.randrange(2 ** 31), spec.path_hops, target)
                generator.apply(scenario)
                time.sleep(args.warmup)
                try:
                    snapshot = link_telemetry.collect()
                    measured = measure_epoch(spec, rng, candidates, args, scenario, counts, quota, installer,
                                             telemetry, snapshot)
                    failures = 0
                except ControllerError as exc:
                    failures += 1
                    if failures >= 3:
                        raise
                    log.warning("Epoch %d skipped (%s); retrying in 30 s", epoch, exc)
                    time.sleep(30)
                    continue
                for src, dst, path, features, label in measured:
                    writer.writerow({"epoch": epoch, "time": round(time.time(), 1), "src": src, "dst": dst,
                                     "path": "-".join(path), "bottleneck_load": round(scenario.path_utilization(path), 4),
                                     **{k: round(v, 4) for k, v in features.items()},
                                     "traffic_class": label, "traffic_class_name": CLASS_NAMES[label]})
                    handle.flush()
                    counts[label] += 1
                log.info("Epoch %d done; rows per class: %s", epoch,
                         ", ".join(f"{CLASS_NAMES[c]} {counts[c]}" for c in range(len(CLASS_NAMES))))
                epoch += 1
    except KeyboardInterrupt:
        log.info("Stopped; rerun the same command to resume")
    finally:
        if generator is not None:
            generator.stop()
        if net is not None:
            try:
                api.clear_flows()
            except ControllerError as exc:
                log.warning("Could not clear static entries: %s", exc)
        stop_network(net)
        chown_to_invoking_user(os.path.dirname(os.path.abspath(args.out)))
    log.info("Dataset: %s (%d rows)", args.out, sum(counts.values()))
    return args.out


def trim(path: str, seed: int = 0) -> str:
    """Cut every class to the size of the smallest one (seeded random sample, original row order kept);
    the untrimmed file is kept as <name>_full.csv."""
    import pandas as pd
    full = pd.read_csv(path)
    n = int(full["traffic_class"].value_counts().min())
    keep = full.groupby("traffic_class", group_keys=False).sample(n=n, random_state=seed).index
    root, ext = os.path.splitext(path)
    full.to_csv(f"{root}_full{ext}", index=False)
    full.loc[sorted(keep)].to_csv(path, index=False)
    log.info("Trimmed %s: %d -> %d rows (%d per class); all rows kept in %s_full%s", path, len(full), 4 * n, n,
             root, ext)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topology", default="geant2", choices=sorted(TOPOLOGIES))
    parser.add_argument("--samples", type=int, default=1000, help="target number of rows (balanced)")
    parser.add_argument("--out", help="default: data/collected/<topology>_dataset.csv")
    parser.add_argument("--extra-pairs", type=int, default=3, help="random host pairs probed per epoch")
    parser.add_argument("--probe-seconds", type=int, default=5, help="iperf3 probe length")
    parser.add_argument("--warmup", type=float, default=5.0, help="seconds after a new scenario before measuring")
    parser.add_argument("--targeted", type=float, default=0.7,
                        help="share of epochs whose scenario targets the class furthest behind its quota")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controller-url", default="http://127.0.0.1:8080")
    parser.add_argument("--trim", metavar="CSV", help="only trim an existing dataset to balanced classes")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("mininet").propagate = False
    if args.trim:
        trim(args.trim)
        return
    args.out = args.out or os.path.join(PROJECT_DIR, "data", "collected", f"{args.topology}_dataset.csv")
    collect(args)


if __name__ == "__main__":
    main()
