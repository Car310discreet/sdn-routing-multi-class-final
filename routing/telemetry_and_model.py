"""
telemetry_and_model.py - Module 3: controller access, telemetry, multi-class inference (Stage 2).

FloodlightClient     REST client (timeouts, retries with backoff, minimum interval between calls)
FlowInstaller        bidirectional static OpenFlow entries for a switch path between two hosts
LinkTelemetry        synchronous link telemetry for Stage 1: polls /wm/topology/links/json and
                     /wm/core/switch/all/port/json `samples` times, `interval_s` apart, and derives per link
                       delay (mean LLDP latency), jitter (std. dev. of the latency samples),
                       available bandwidth (capacity - measured rate, busier direction),
                       loss (1 - far-end rx packets / near-end tx packets, worse direction)
                     No background threads: one call, one snapshot.
PathTelemetry        the 11 features of the GEANT2 multi-class dataset for one candidate path:
                       ping -c 50 -i 0.05 -W 5     rtt avg/min/max/mdev, packet loss
                       iperf3 TCP 5 s              throughput, bandwidth utilisation, retransmits
                       path length                 hop count
                       LinkTelemetry snapshot      max / mean link utilisation along the path
SoftVotingEnsemble   averages the class probabilities of several fitted classifiers
MultiClassModel      loads the bundle written by train_models.py (ensemble or a single model)
MultiClassPathSelector
                     Stage 2: pins each candidate's ICMP/TCP, measures its features, predicts its
                     congestion class (0 Extremely Low ... 3 Extremely High), picks the lowest class
                     (ties: lower Stage-1 composite cost, then Stage-1 rank) and installs it.
"""
from __future__ import annotations

import json
import logging
import os
import re
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field

import networkx as nx
import numpy as np
import requests

log = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# each topology's model, trained on its own dataset (routing/train_models.py)
MODEL_PATHS = {t: os.path.join(PROJECT_DIR, "models", f"{t}_bundle.joblib") for t in ("geant2", "fattree")}
DEFAULT_MODEL = MODEL_PATHS["geant2"]
ACTIVE_FEATURES = ["rtt_avg_ms", "rtt_min_ms", "rtt_max_ms", "rtt_mdev_ms", "packet_loss_pct",
                   "throughput_mbps", "bw_utilization_pct", "retransmits", "hop_count"]
PASSIVE_FEATURES = ["path_max_link_util_pct", "path_mean_link_util_pct"]
FEATURE_NAMES = ACTIVE_FEATURES + PASSIVE_FEATURES
CLASS_NAMES = ["Extremely Low", "Low", "High", "Extremely High"]
PROBE_PORT = 5401
PROBE_PRIORITY = 40000
ROUTE_PRIORITY = 32768


class ControllerError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------
# Floodlight
# ----------------------------------------------------------------------------------------
class FloodlightClient:
    def __init__(self, base_url="http://127.0.0.1:8080", timeout=10.0, retries=3, min_interval=0.02):
        self.base_url, self.timeout, self.retries, self.min_interval = base_url.rstrip("/"), timeout, retries, min_interval
        self.session = requests.Session()
        self._last = 0.0

    def _request(self, method: str, path: str, **kwargs):
        delay, error = 0.5, None
        for _ in range(self.retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                response = self.session.request(method, self.base_url + path, timeout=self.timeout, **kwargs)
                response.raise_for_status()
                return response.json() if response.content else {}
            except (requests.RequestException, ValueError) as exc:
                error = exc
                time.sleep(delay)
                delay *= 2
        raise ControllerError(f"{method} {path} failed after {self.retries} attempts: {error}")

    def health(self) -> bool:
        try:
            return bool(self._request("GET", "/wm/core/health/json").get("healthy"))
        except ControllerError:
            return False

    def switches(self) -> list[dict]:
        return self._request("GET", "/wm/core/controller/switches/json")

    def links(self) -> list[dict]:
        return self._request("GET", "/wm/topology/links/json")

    def port_counters(self) -> dict[tuple[str, int], dict]:
        """(DPID, port) -> transmit/receive bytes and packets, for every switch port."""
        counters = {}
        for dpid, reply in self._request("GET", "/wm/core/switch/all/port/json").items():
            if not isinstance(reply, dict):
                continue
            for part in reply.get("port_reply", []):
                for entry in part.get("port", []):
                    try:
                        counters[(dpid, int(entry["port_number"]))] = {
                            key: int(entry[key]) for key in
                            ("transmit_bytes", "receive_bytes", "transmit_packets", "receive_packets")}
                    except (KeyError, TypeError, ValueError):
                        continue             # the LOCAL port, or a malformed entry
        return counters

    def set_routing_metric(self, metric: str) -> None:
        self._request("POST", "/wm/routing/metric/json", json={"metric": metric})

    def push_flow(self, flow: dict) -> None:
        result = self._request("POST", "/wm/staticentrypusher/json", json=flow)
        if "pushed" not in str(result.get("status", "")).lower():
            raise ControllerError(f"Flow {flow.get('name')} rejected: {result}")

    def delete_flow(self, name: str) -> None:
        self._request("DELETE", "/wm/staticentrypusher/json", data=json.dumps({"name": name}))

    def clear_flows(self) -> None:
        self._request("GET", "/wm/staticentrypusher/clear/all/json")


def wait_for_switches(client: FloodlightClient, dpids: set[str], timeout_s=180, stable_s=10) -> None:
    """Block until every switch is connected and has stayed connected for `stable_s` seconds."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            switches = {s.get("switchDPID"): s for s in client.switches()}
            if dpids <= set(switches):
                now_ms = time.time() * 1000
                if min(now_ms - float(switches[d]["connectedSince"]) for d in dpids) >= stable_s * 1000:
                    return
        except ControllerError as exc:
            log.debug("Controller not answering yet: %s", exc)
        time.sleep(2)
    raise ControllerError(f"Switches did not connect within {timeout_s}s")


class FlowInstaller:
    """Pushes / removes bidirectional static entries for a switch path between two hosts."""

    def __init__(self, client: FloodlightClient, spec):
        self.client, self.spec = client, spec

    def install_path(self, src_ip: str, dst_ip: str, path: list[str], prefix: str, priority: int = ROUTE_PRIORITY,
                     ip_protos=None) -> list[str]:
        src, dst = self.spec.host_by_ip(src_ip), self.spec.host_by_ip(dst_ip)
        if path[0] != src.switch or path[-1] != dst.switch:
            raise ValueError(f"Path {path} does not connect {src.name} to {dst.name}")
        names = []
        try:
            for i, switch in enumerate(path):
                in_port = src.switch_port if i == 0 else self.spec.port(switch, path[i - 1])
                out_port = dst.switch_port if i == len(path) - 1 else self.spec.port(switch, path[i + 1])
                for proto in (ip_protos or [None]):
                    for direction, (a, b, p_in, p_out) in enumerate(
                            ((src_ip, dst_ip, in_port, out_port), (dst_ip, src_ip, out_port, in_port))):
                        name = f"{prefix}_{switch}_{direction}" + ("" if proto is None else f"_p{proto}")
                        flow = {"switch": self.spec.switch_dpids[switch], "name": name, "priority": str(priority),
                                "eth_type": "0x0800", "ipv4_src": a, "ipv4_dst": b, "in_port": str(p_in),
                                "active": "true", "actions": f"output={p_out}"}
                        if proto is not None:
                            flow["ip_proto"] = str(proto)
                        self.client.push_flow(flow)
                        names.append(name)
        except Exception:
            self.remove(names)       # never leave a half-installed path behind
            raise
        return names

    def remove(self, names: list[str]) -> None:
        for name in names:
            try:
                self.client.delete_flow(name)
            except ControllerError as exc:
                log.warning("Could not delete flow %s: %s", name, exc)


# ----------------------------------------------------------------------------------------
# Link telemetry (Stage 1 input; also the passive Stage-2 features)
# ----------------------------------------------------------------------------------------
@dataclass
class LinkSnapshot:
    window_s: float
    collect_s: float
    rate_mbps: dict = field(default_factory=dict)      # directed (u, v) -> Mbit/s sent u -> v
    stats: dict = field(default_factory=dict)          # link_key -> {delay_ms, jitter_ms, avail_bw_mbps, loss_rate}

    def annotated_graph(self, graph: nx.Graph) -> nx.Graph:
        annotated = graph.copy()
        for u, v, data in annotated.edges(data=True):
            data.update(self.stats.get(tuple(sorted((u, v), key=lambda s: int(s[1:]))), {}))
        return annotated

    def utilization(self, u: str, v: str, capacity_mbps: float) -> float:
        return self.rate_mbps.get((u, v), 0.0) / capacity_mbps


def link_stats_from(spec, latency_samples: dict, first: dict, last: dict, seconds: float):
    """Per-link metrics from LLDP latency samples and two port-counter snapshots `seconds` apart."""
    rate, loss = {}, {}
    for a, b in spec.graph.edges:
        for u, v in ((a, b), (b, a)):
            near = (spec.switch_dpids[u], spec.port(u, v))
            far = (spec.switch_dpids[v], spec.port(v, u))
            if near in first and near in last:
                sent_bytes = last[near]["transmit_bytes"] - first[near]["transmit_bytes"]
                rate[(u, v)] = max(sent_bytes, 0) * 8 / max(seconds, 1e-3) / 1e6
                sent = last[near]["transmit_packets"] - first[near]["transmit_packets"]
                if sent > 0 and far in first and far in last:
                    got = last[far]["receive_packets"] - first[far]["receive_packets"]
                    loss[(u, v)] = min(1.0, max(0.0, 1 - got / sent))
    stats = {}
    for u, v, data in spec.graph.edges(data=True):
        key = tuple(sorted((u, v), key=lambda s: int(s[1:])))
        samples = latency_samples.get(key, [])
        used = max(rate.get((u, v), 0.0), rate.get((v, u), 0.0))
        stats[key] = {
            "delay_ms": round(statistics.fmean(samples), 3) if samples else data.get("delay_ms", 0.0),
            "jitter_ms": round(statistics.pstdev(samples), 3) if len(samples) > 1 else 0.0,
            "avail_bw_mbps": round(max(0.0, data.get("capacity_mbps", spec.link_bw_mbps) - used), 4),
            "loss_rate": round(max(loss.get((u, v), 0.0), loss.get((v, u), 0.0)), 4)}
    return rate, stats


class LinkTelemetry:
    def __init__(self, api: FloodlightClient, spec, samples: int = 6, interval_s: float = 2.5):
        if samples < 2 or interval_s < 0.5:
            raise ValueError("need >= 2 samples at least 0.5 s apart")
        self.api, self.spec, self.samples, self.interval_s = api, spec, samples, interval_s

    def collect(self) -> LinkSnapshot:
        started = time.monotonic()
        names = self.spec.dpid_to_switch
        latency: dict[tuple, list] = {}
        first = last = None
        t_first = t_last = None
        for i in range(self.samples):
            if i:
                time.sleep(self.interval_s)
            for link in self.api.links():
                u, v = names.get(link.get("src-switch")), names.get(link.get("dst-switch"))
                if u and v and self.spec.graph.has_edge(u, v) and link.get("latency") is not None:
                    key = tuple(sorted((u, v), key=lambda s: int(s[1:])))
                    latency.setdefault(key, []).append(float(link["latency"]))
            counters, now = self.api.port_counters(), time.monotonic()
            if first is None:
                first, t_first = counters, now
            last, t_last = counters, now
        rate, stats = link_stats_from(self.spec, latency, first, last, t_last - t_first)
        return LinkSnapshot(round(t_last - t_first, 3), round(time.monotonic() - started, 3), rate, stats)


# ----------------------------------------------------------------------------------------
# Candidate-path features (the GEANT2 multi-class dataset's 11 features)
# ----------------------------------------------------------------------------------------
def parse_ping(output: str, deadline_s: float) -> dict:
    loss = re.search(r"([\d.]+)% packet loss", output)
    rtt = re.search(r"= ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms", output)
    if rtt:
        rtt_min, rtt_avg, rtt_max, rtt_mdev = (float(x) for x in rtt.groups())
    else:                                  # nothing came back: as slow as the deadline allows
        rtt_min = rtt_avg = rtt_max = deadline_s * 1000.0
        rtt_mdev = 0.0
    return {"rtt_avg_ms": rtt_avg, "rtt_min_ms": rtt_min, "rtt_max_ms": rtt_max, "rtt_mdev_ms": rtt_mdev,
            "packet_loss_pct": float(loss.group(1)) if loss else 100.0}


def parse_iperf(stdout: str, capacity_mbps: float) -> dict:
    try:
        end = json.loads(stdout)["end"]
    except (ValueError, KeyError, TypeError):
        end = {}
    received, sent = end.get("sum_received") or {}, end.get("sum_sent") or {}
    throughput = float(received.get("bits_per_second", 0.0)) / 1e6
    return {"throughput_mbps": throughput,
            "bw_utilization_pct": min(max(100.0 * (1.0 - throughput / capacity_mbps), 0.0), 100.0),
            "retransmits": int(sent.get("retransmits", 0))}


def passive_features(snapshot: LinkSnapshot, path: list[str], capacity_mbps: float) -> dict:
    values = [snapshot.utilization(a, b, capacity_mbps) for u, v in zip(path, path[1:]) for a, b in ((u, v), (v, u))]
    return {"path_max_link_util_pct": 100.0 * max(values), "path_mean_link_util_pct": 100.0 * sum(values) / len(values)}


class PathTelemetry:
    """Active probe of one path, as in the GEANT2 dataset collection (same commands, same durations)."""

    def __init__(self, net, spec, ping_count=50, ping_interval=0.05, ping_deadline_s=5.0, probe_seconds=5,
                 drain_s=2.0, port=PROBE_PORT):
        self.net, self.spec = net, spec
        self.ping_count, self.ping_interval, self.ping_deadline_s = ping_count, ping_interval, ping_deadline_s
        self.probe_seconds, self.drain_s, self.port = probe_seconds, drain_s, port
        self.last_timing: dict = {}      # seconds spent in each step of the last measure()

    def measure(self, src_host: str, dst_host: str, path: list[str], snapshot: LinkSnapshot) -> dict:
        src, dst = self.net.get(src_host), self.net.get(dst_host)
        dst_ip = self.spec.hosts[dst_host].ip
        t0 = time.monotonic()
        features = parse_ping(src.cmd(f"ping -q -c {self.ping_count} -i {self.ping_interval} "
                                      f"-W {self.ping_deadline_s:g} {dst_ip}"), self.ping_deadline_s)
        t1 = time.monotonic()
        server = dst.popen(["iperf3", "-s", "-1", "-p", str(self.port)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        time.sleep(0.5)
        client = src.popen(["iperf3", "-c", dst_ip, "-p", str(self.port), "-t", str(self.probe_seconds), "-J"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        try:
            stdout, _ = client.communicate(timeout=self.probe_seconds + 30)
        except subprocess.TimeoutExpired:
            client.kill()
            stdout = ""
        if server.poll() is None:
            server.terminate()
        features.update(parse_iperf(stdout, self.spec.link_bw_mbps))
        t2 = time.monotonic()
        time.sleep(self.drain_s)         # let the probe's queues drain before the next path is measured
        t3 = time.monotonic()
        features["hop_count"] = len(path) - 1
        features.update(passive_features(snapshot, path, self.spec.link_bw_mbps))
        self.last_timing = {"ping_s": t1 - t0, "iperf_s": t2 - t1, "drain_s": t3 - t2}
        return {name: float(features[name]) for name in FEATURE_NAMES}


# ----------------------------------------------------------------------------------------
# Multi-class model
# ----------------------------------------------------------------------------------------
class SoftVotingEnsemble:
    """Average of the class probabilities of already fitted classifiers (classes 0..n-1)."""

    def __init__(self, members: dict):
        if not members:
            raise ValueError("Ensemble needs at least one member")
        self.members = members
        self.classes_ = np.arange(len(CLASS_NAMES))

    def predict_proba(self, X) -> np.ndarray:
        return np.mean([m.predict_proba(X) for m in self.members.values()], axis=0)

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class MultiClassModel:
    def __init__(self, bundle: dict, which: str = "ensemble"):
        options = {"ensemble": bundle["ensemble"], **bundle["models"]}
        if which not in options:
            raise ValueError(f"Unknown model {which!r}; bundle has {sorted(options)}")
        self.name, self.model = which, options[which]
        self.features, self.classes = list(bundle["features"]), list(bundle["classes"])
        self.last_assembly_ms = 0.0          # feature-matrix build time of the last predict_batch()

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL, which: str = "ensemble") -> "MultiClassModel":
        import joblib
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model bundle not found: {path} (run routing/train_models.py)")
        return cls(joblib.load(path), which)

    def predict_batch(self, feature_rows: list[dict]) -> tuple[list[int], np.ndarray, float]:
        """(classes 0-3, class probabilities, inference time in ms) for all candidates in one call.
        One batched call pays each member model's fixed per-call overhead once instead of K times."""
        assembly_start = time.perf_counter()
        rows = np.asarray([[f[name] for name in self.features] for f in feature_rows], dtype=float)
        if not np.all(np.isfinite(rows)):
            raise ValueError(f"Non-finite feature vector in {feature_rows}")
        start = time.perf_counter()
        probabilities = self.model.predict_proba(rows)
        inference_ms = (time.perf_counter() - start) * 1000
        self.last_assembly_ms = (start - assembly_start) * 1000
        return [int(c) for c in np.argmax(probabilities, axis=1)], probabilities, inference_ms

    def predict(self, features: dict) -> tuple[int, np.ndarray, float]:
        classes, probabilities, ms = self.predict_batch([features])
        return classes[0], probabilities[0], ms


# ----------------------------------------------------------------------------------------
# Stage 2
# ----------------------------------------------------------------------------------------
@dataclass
class ScoredCandidate:
    rank_in_stage1: int
    switches: list[str]
    composite_cost: float
    features: dict
    predicted_class: int
    predicted_name: str
    probabilities: list[float]
    probe_s: float
    timing: dict = field(default_factory=dict)     # this candidate's probe steps (see Stage2Result.timing)


@dataclass
class Stage2Result:
    model: str
    chosen_rank: int
    chosen: list[str]
    probe_s: float
    inference_ms: float
    scored: list[ScoredCandidate] = field(default_factory=list)
    # totals over all candidates: probe_install_ms / probe_remove_ms (static entries pinning the probe),
    # ping_s, iperf_s, drain_s (PathTelemetry), assembly_ms (feature matrix), inference_ms (one batched
    # predict_proba for all K candidates), select_ms (choose())
    timing: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def choose(scored: list[ScoredCandidate]) -> ScoredCandidate:
    """Lowest predicted class; ties to the lower Stage-1 composite cost, then the earlier Stage-1 rank."""
    return min(scored, key=lambda s: (s.predicted_class, s.composite_cost, s.rank_in_stage1))


class MultiClassPathSelector:
    def __init__(self, installer: FlowInstaller, telemetry: PathTelemetry, model: MultiClassModel):
        self.installer, self.telemetry, self.model = installer, telemetry, model

    def select(self, candidates, src_host: str, dst_host: str, snapshot: LinkSnapshot) -> Stage2Result:
        """Score every Stage-1 candidate (`CandidatePath`s); does not install the winner."""
        if not candidates:
            raise ValueError("No candidate paths")
        spec = self.telemetry.spec
        src_ip, dst_ip = spec.hosts[src_host].ip, spec.hosts[dst_host].ip
        measured = []
        for rank, candidate in enumerate(candidates):
            t0 = time.monotonic()
            names = self.installer.install_path(src_ip, dst_ip, candidate.switches, prefix=f"probe{rank}",
                                                priority=PROBE_PRIORITY, ip_protos=[1, 6])
            started = time.monotonic()
            try:
                features = self.telemetry.measure(src_host, dst_host, candidate.switches, snapshot)
            finally:
                measured_at = time.monotonic()
                self.installer.remove(names)
            steps = {"probe_install_ms": (started - t0) * 1000, "probe_remove_ms": (time.monotonic() - measured_at) * 1000,
                     **getattr(self.telemetry, "last_timing", {})}
            measured.append((features, round(measured_at - started, 2), steps))
        # every candidate's features in one batched predict_proba call
        classes, probabilities, inference_ms = self.model.predict_batch([f for f, _, _ in measured])
        scored = []
        for rank, (candidate, (features, probe_s, steps), predicted, probs) in enumerate(
                zip(candidates, measured, classes, probabilities)):
            scored.append(ScoredCandidate(rank, candidate.switches, candidate.composite_cost,
                                          {k: round(v, 4) for k, v in features.items()}, predicted,
                                          CLASS_NAMES[predicted], [round(float(p), 4) for p in probs], probe_s,
                                          {k: round(v, 4) for k, v in steps.items()}))
            log.info("  candidate %d %s: %s (composite %.3f, %.2f Mbit/s, rtt %.0f ms)", rank,
                     "-".join(candidate.switches), CLASS_NAMES[predicted], candidate.composite_cost,
                     features["throughput_mbps"], features["rtt_avg_ms"])
        select_start = time.perf_counter()
        best = choose(scored)
        select_ms = (time.perf_counter() - select_start) * 1000
        timing = {key: round(sum(s.timing.get(key, 0.0) for s in scored), 4)
                  for key in ("probe_install_ms", "probe_remove_ms", "ping_s", "iperf_s", "drain_s")}
        timing.update(assembly_ms=round(getattr(self.model, "last_assembly_ms", 0.0), 4),
                      inference_ms=round(inference_ms, 4), select_ms=round(select_ms, 4))
        return Stage2Result(self.model.name, best.rank_in_stage1, best.switches,
                            round(sum(s.probe_s for s in scored), 2), round(inference_ms, 4), scored, timing)
