"""Offline tests of the routing modules (run: .venv/bin/python -m pytest tests).

No Mininet or Floodlight: controller replies, telemetry and run results are synthetic."""
import itertools
import json
import os
import sys
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "routing"))
import multimedia_evaluator  # noqa: E402
import udp_probe  # noqa: E402
import visualize_benchmarks  # noqa: E402
from path_discovery import (METHODS, CandidatePath, composite_link_costs, discover_paths,  # noqa: E402
                            synthetic_link_metrics)
from telemetry_and_model import (DEFAULT_MODEL, FEATURE_NAMES, MultiClassModel, MultiClassPathSelector,  # noqa: E402
                                 ScoredCandidate, choose, link_stats_from, parse_iperf, parse_ping)
from topology_builder import (CLASS_BANDS, RECEIVER, SENDER, build_fattree, build_geant2, class_of,  # noqa: E402
                              link_class_weights, sample_scenario)

SPEC = build_geant2()
SRC, DST = SPEC.hosts[SENDER].switch, SPEC.hosts[RECEIVER].switch


# -- Module 1 --------------------------------------------------------------------------------
def test_geant2_topology():
    assert SPEC.graph.number_of_nodes() == 24 and SPEC.graph.number_of_edges() == 37
    assert nx.shortest_path_length(SPEC.graph, SRC, DST) == nx.diameter(SPEC.graph) == 6
    for switch in SPEC.graph.nodes:
        ports = [SPEC.port(switch, n) for n in SPEC.graph.neighbors(switch)]
        ports += [h.switch_port for h in SPEC.hosts.values() if h.switch == switch]
        assert len(ports) == len(set(ports))
    assert set(nx.get_edge_attributes(SPEC.graph, "delay_ms").values()) == {10.0}


def test_fattree_topology():
    spec = build_fattree()
    assert spec.graph.number_of_nodes() == 20 and spec.graph.number_of_edges() == 32
    assert (spec.sender, spec.receiver, spec.path_hops) == ("h12", "h18", 4.0)
    src, dst = spec.hosts[spec.sender].switch, spec.hosts[spec.receiver].switch
    assert nx.shortest_path_length(spec.graph, src, dst) == nx.diameter(spec.graph) == 4
    hops = [p.hops for p in discover_paths(spec.graph, src, dst, "hop", 5).paths]
    assert hops == [4, 4, 4, 4, 6]                      # four equal-cost paths, one per core switch
    counts = Counter(sample_scenario(spec.graph, seed, spec.path_hops).path_class(p.switches)
                     for seed in range(500) for p in discover_paths(spec.graph, src, dst, "hop", 5).paths[:4])
    assert all(0.18 < counts[c] / sum(counts.values()) < 0.32 for c in range(4))


def test_four_class_traffic():
    assert [class_of(x) for x in (0.0, 0.24, 0.25, 0.6, 0.8, 1.19)] == [0, 0, 1, 2, 3, 3]
    assert np.isclose(link_class_weights().sum(), 1.0)
    hop = [p.switches for p in discover_paths(SPEC.graph, SRC, DST, "hop", 5).paths]
    counts = Counter(sample_scenario(SPEC.graph, s).path_class(p) for s in range(400) for p in hop)
    assert all(0.15 < counts[c] / sum(counts.values()) < 0.35 for c in range(4))
    scenario = sample_scenario(SPEC.graph, 1)
    scenario.congest(hop[0])
    assert scenario.path_class(hop[0]) == 3


# -- Module 2 --------------------------------------------------------------------------------
GRAPH = synthetic_link_metrics(SPEC.graph, seed=7)


def bounded_simple_paths(max_hops):
    return list(nx.all_simple_paths(GRAPH, SRC, DST, cutoff=max_hops))


def test_hop_and_latency_match_brute_force():
    hop = discover_paths(GRAPH, SRC, DST, "hop", 5)
    paths = bounded_simple_paths(hop.max_hops)
    assert [p.hops for p in hop.paths] == sorted(len(p) - 1 for p in paths)[:5]
    lat = discover_paths(GRAPH, SRC, DST, "latency", 5)
    delay = lambda p: sum(GRAPH.edges[e]["delay_ms"] for e in zip(p, p[1:]))  # noqa: E731
    assert [round(p.cost, 3) for p in lat.paths] == [round(d, 3) for d in sorted(map(delay, paths))[:5]]


def test_widest_and_composite_match_brute_force():
    paths = bounded_simple_paths(discover_paths(GRAPH, SRC, DST, "hop", 1).max_hops)
    width = lambda p: min(GRAPH.edges[e]["avail_bw_mbps"] for e in zip(p, p[1:]))  # noqa: E731
    widest = discover_paths(GRAPH, SRC, DST, "bandwidth", 5)
    assert widest.paths[0].bottleneck_mbps == pytest.approx(max(map(width, paths)), abs=1e-3)
    costs = composite_link_costs(GRAPH)
    cost = lambda p: sum(costs[e] for e in zip(p, p[1:]))  # noqa: E731
    composite = discover_paths(GRAPH, SRC, DST, "composite", 5)
    assert [round(p.cost, 4) for p in composite.paths] == [round(c, 4) for c in sorted(map(cost, paths))[:5]]


def test_every_method_reports_time_and_composite_cost():
    costs = composite_link_costs(GRAPH)
    for method, k in itertools.product(METHODS, (5, 10)):
        result = discover_paths(GRAPH, SRC, DST, method, k)
        assert len(result.paths) == k and result.compute_ms > 0
        for p in result.paths:
            assert p.composite_cost == pytest.approx(sum(costs[e] for e in zip(p.switches, p.switches[1:])), abs=1e-4)


def test_composite_cost_terms():
    g = nx.Graph()
    g.add_edge("s0", "s1", delay_ms=10, jitter_ms=2, avail_bw_mbps=10, loss_rate=0.0)
    g.add_edge("s1", "s2", delay_ms=5, jitter_ms=0, avail_bw_mbps=0.0, loss_rate=0.5)
    c = composite_link_costs(g, alpha=0.3, beta=0.1, gamma=0.4, delta=0.2)
    assert c[("s0", "s1")] == pytest.approx(0.3 + 0.1 + 0.4)
    assert c[("s1", "s2")] == pytest.approx(0.15 + 0 + 0.4 / 1e-3 + 0.1)


# -- Module 3 --------------------------------------------------------------------------------
def test_link_stats_from_counters():
    u, v = "s15", "s14"
    near, far = (SPEC.switch_dpids[u], SPEC.port(u, v)), (SPEC.switch_dpids[v], SPEC.port(v, u))
    zero = {"transmit_bytes": 0, "receive_bytes": 0, "transmit_packets": 0, "receive_packets": 0}
    first = {near: dict(zero), far: dict(zero)}
    last = {near: {**zero, "transmit_bytes": 5_000_000, "transmit_packets": 1000},
            far: {**zero, "receive_packets": 900}}
    rate, stats = link_stats_from(SPEC, {("s14", "s15"): [20.0, 22.0, 24.0]}, first, last, 8.0)
    assert rate[(u, v)] == pytest.approx(5.0)
    s = stats[("s14", "s15")]
    assert s["avail_bw_mbps"] == pytest.approx(5.0) and s["loss_rate"] == pytest.approx(0.1)
    assert s["delay_ms"] == pytest.approx(22.0) and s["jitter_ms"] == pytest.approx(np.std([20, 22, 24]), abs=1e-3)


def test_parsers():
    out = "50 packets transmitted, 47 received, 6% packet loss\nrtt min/avg/max/mdev = 61.1/80.2/130.9/9.3 ms"
    assert parse_ping(out, 5)["rtt_avg_ms"] == 80.2 and parse_ping("100% packet loss", 5)["rtt_avg_ms"] == 5000
    iperf = json.dumps({"end": {"sum_received": {"bits_per_second": 4e6}, "sum_sent": {"retransmits": 7}}})
    assert parse_iperf(iperf, 10) == {"throughput_mbps": 4.0, "bw_utilization_pct": 60.0, "retransmits": 7}


requires_model = pytest.mark.skipif(not os.path.exists(DEFAULT_MODEL), reason="run routing/train_models.py first")


@requires_model
def test_ensemble_bundle_predicts_four_classes():
    model = MultiClassModel.load()
    assert model.features == FEATURE_NAMES and len(model.classes) == 4
    rows = pd.read_csv(os.path.join(os.path.dirname(DEFAULT_MODEL), "..", "data", "geant2_dataset.csv"))
    sample = rows.groupby("traffic_class").head(3)
    classes, probs, ms = model.predict_batch(sample[FEATURE_NAMES].to_dict("records"))
    assert probs.shape == (12, 4) and np.allclose(probs.sum(axis=1), 1) and ms > 0
    assert np.mean(np.array(classes) == sample["traffic_class"].to_numpy()) >= 0.9


def test_choose_breaks_ties_on_composite_cost():
    mk = lambda rank, cls, cost: ScoredCandidate(rank, [f"s{rank}"], cost, {}, cls, "", [], 0.0)  # noqa: E731
    assert choose([mk(0, 1, 5.0), mk(1, 0, 9.0), mk(2, 0, 3.0)]).rank_in_stage1 == 2
    assert choose([mk(0, 0, 3.0), mk(1, 0, 3.0)]).rank_in_stage1 == 0


class FakeInstaller:
    def __init__(self):
        self.calls = []

    def install_path(self, src_ip, dst_ip, path, prefix, priority=0, ip_protos=None):
        self.calls.append((tuple(path), ip_protos))
        return [prefix]

    def remove(self, names):
        pass


class FakeTelemetry:
    spec = SPEC

    def __init__(self, loads):
        self.loads = loads

    def measure(self, src, dst, path, snapshot):
        load = self.loads[tuple(path)]
        thr = max(10 * (1 - load), 0.2)
        rtt = 120 + 2400 * max(load - 0.9, 0) * 4
        return {"rtt_avg_ms": rtt, "rtt_min_ms": rtt, "rtt_max_ms": rtt * 1.2, "rtt_mdev_ms": 0.4,
                "packet_loss_pct": 70.0 if load > 1 else 0.0, "throughput_mbps": thr if load < 1 else 0.0,
                "bw_utilization_pct": 100 * (1 - thr / 10), "retransmits": 0.0, "hop_count": len(path) - 1.0,
                "path_max_link_util_pct": 100 * min(load, 1.0), "path_mean_link_util_pct": 40 * load}


@requires_model
def test_selector_picks_least_congested_candidate():
    candidates = discover_paths(GRAPH, SRC, DST, "composite", 5).paths
    loads = dict(zip((tuple(c.switches) for c in candidates), (1.1, 0.7, 0.1, 0.4, 0.9)))
    installer = FakeInstaller()
    result = MultiClassPathSelector(installer, FakeTelemetry(loads), MultiClassModel.load()).select(
        candidates, SENDER, RECEIVER, snapshot=None)
    assert result.chosen == candidates[2].switches and result.scored[2].predicted_class == 0
    assert all(protos == [1, 6] for _, protos in installer.calls)        # probes pin ICMP/TCP only
    assert {"probe_install_ms", "ping_s", "assembly_ms", "inference_ms", "select_ms"} <= set(result.timing)


@requires_model
def test_stage2_predicts_all_candidates_in_one_batched_call():
    candidates = discover_paths(GRAPH, SRC, DST, "composite", 5).paths
    loads = dict(zip((tuple(c.switches) for c in candidates), (1.1, 0.7, 0.1, 0.4, 0.9)))
    model = MultiClassModel.load()
    calls = []
    original = model.predict_batch
    model.predict_batch = lambda rows: calls.append(len(rows)) or original(rows)
    MultiClassPathSelector(FakeInstaller(), FakeTelemetry(loads), model).select(candidates, SENDER, RECEIVER, None)
    assert calls == [5]


# -- Modules 4 and 5 ---------------------------------------------------------------------------
def test_udp_probe_loss_and_jitter():
    sent = [{"seq": i, "send_ts": 100 + i * 0.1} for i in range(20)]
    recv = [{"seq": i, "send_ts": 100 + i * 0.1, "recv_ts": 100 + i * 0.1 + 0.05} for i in range(20) if i != 3]
    stats = udp_probe.per_second(sent, recv, 100.0, 2)
    assert stats[1]["loss_pct"] == pytest.approx(10.0) and stats[2]["loss_pct"] == 0.0
    assert stats[2]["jitter_ms"] == pytest.approx(0.0, abs=1e-6)


def test_route_counters_exclude_probe(tmp_path):
    dump = (" cookie=0x0, duration=5s, table=0, n_packets=120, n_bytes=150000, priority=32768,ip,in_port=3 actions=output:2\n"
            " cookie=0x0, duration=9s, table=0, n_packets=5, n_bytes=490, priority=1,ip actions=CONTROLLER:65535\n")
    assert multimedia_evaluator.parse_flow_counters(dump) == (125, 150490)

    start = 1000.0
    sent, recv = tmp_path / "sent.csv", tmp_path / "recv.csv"
    sent.write_text("seq,send_ts\n" + "".join(f"{i},{start + i * 0.5:.6f}\n" for i in range(4)))
    recv.write_text("seq,send_ts,recv_ts\n" + "".join(f"{i},{start + i * 0.5:.6f},{start + i * 0.5 + 0.1:.6f}\n"
                                                      for i in (0, 1, 3)))
    rows = [{"t_s": 1, "throughput_mbps": 1.0}, {"t_s": 2, "throughput_mbps": 1.0}]
    totals = {"sent": 100, "received": 90}
    multimedia_evaluator.VideoEvaluator._merge_probe(rows, totals, str(sent), str(recv), start)
    assert totals == {"sent": 96, "received": 87}
    frame_mbit = multimedia_evaluator.PROBE_FRAME_BYTES * 8 / 1e6
    assert rows[0]["throughput_mbps"] == pytest.approx(1.0 - 2 * frame_mbit, abs=1e-4)
    assert rows[1]["throughput_mbps"] == pytest.approx(1.0 - frame_mbit, abs=1e-4)
    assert rows[1]["packet_loss_pct"] == 50.0


def test_visualize_writes_all_figures(tmp_path):
    rng = np.random.default_rng(0)
    for k in range(2):
        for i, paradigm in enumerate(visualize_benchmarks.PARADIGMS):
            d = tmp_path / "geant2_x" / f"scenario_{k}" / paradigm
            d.mkdir(parents=True)
            json.dump({"psnr_db": 20 + 5 * i, "ssim": 0.8 + 0.05 * i, "mean_rtt_ms": 900 - 200 * i,
                       "mean_jitter_ms": 20 - 4 * i, "mean_throughput_mbps": 3 + i * 0.5, "packet_loss_pct": 6 - 2 * i,
                       "frames_decoded": 4000}, open(d / "summary.json", "w"))
            pd.DataFrame({"t_s": range(1, 61), "rtt_ms": rng.normal(900 - 200 * i, 20, 60),
                          "jitter_ms": rng.normal(20 - 4 * i, 2, 60), "throughput_mbps": rng.normal(3 + i * 0.5, 0.2, 60),
                          "packet_loss_pct": np.clip(rng.normal(6 - 2 * i, 2, 60), 0, None)}).to_csv(
                d / "timeseries.csv", index=False)
            json.dump({"chosen_path": ["s15", "s23"], "chosen_ground_truth": {"class": 3 - i, "class_name": "x"},
                       "timing": {"stage1_ms": 1.0 + i, "stage2_inference_ms": 25.0 * (i >= 2),
                                  "telemetry_s": 12.5, "stage2_probe_s": 60.0 * (i >= 2)}},
                      open(d / "decision.json", "w"))
    bench = tmp_path / "bench.csv"
    pd.DataFrame([{"trial": 0, "method": m, "k": k, "stage1_ms": 2.0, "stage2_inference_ms": 28.0}
                  for m in METHODS for k in (5, 10)]).to_csv(bench, index=False)
    summary = visualize_benchmarks.visualize(str(tmp_path / "geant2_x"), str(bench))
    assert summary.loc["Ours: combined + ML", "psnr_db"] == pytest.approx(5.0 * 4 + 20)
    for name in ("timeseries_scenario_0.png", "timeseries_scenario_1.png", "packet_loss_boxplot.png",
                 "video_quality.png", "computational_latency.png", "computational_latency_offline.png",
                 "summary_table.csv", "per_scenario.csv"):
        assert (tmp_path / "geant2_x" / "figures" / name).exists()
