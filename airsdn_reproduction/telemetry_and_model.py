"""
telemetry_and_model.py - Stage 2 of AIRSDN: path statistics, Logistic Regression, flow rules.

Implements the paper's routing algorithm (Table 2):
    for each of the 5 candidate paths:
        AddToFlowTable(path)                    -> FlowInstaller (Floodlight StaticEntryPusher)
        pathStatistics <- GetThePathStatistics() -> PathProber (19 features: ping, iperf3, NFStream)
        prediction <- model.Prediction(...)      -> TrafficClassifier (Logistic Regression)
        DeleteAddedFlows(path)
    path <- FindThePathWithLowestTraffic(...)   -> lowest predicted traffic level, ties -> first
    AddToFlowTable(path)

Probe entries match only ICMP and TCP (higher priority), so probing never redirects other traffic.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import requests

log = logging.getLogger(__name__)

# Feature order of data/result_data.csv (the order the model was trained on)
FEATURE_NAMES = [
    "average_rtt", "packet_loss", "bits_per_second", "bu_ratio", "retransmits", "cpu_host_total",
    "cpu_host_system", "bidirectional_duration_ms", "src2dst_duration_ms", "dst2src_duration_ms",
    "bidirectional_mean_piat_ms", "bidirectional_stddev_piat_ms", "bidirectional_max_piat_ms",
    "src2dst_mean_piat_ms", "src2dst_stddev_piat_ms", "src2dst_max_piat_ms",
    "dst2src_mean_piat_ms", "dst2src_stddev_piat_ms", "dst2src_max_piat_ms",
]
NFSTREAM_FEATURES = FEATURE_NAMES[7:]


class ControllerError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------
# Floodlight REST API
# ----------------------------------------------------------------------------------------
class FloodlightClient:
    def __init__(self, base_url="http://127.0.0.1:8080", timeout=5.0, retries=3):
        self.base_url, self.timeout, self.retries = base_url.rstrip("/"), timeout, retries
        self.session = requests.Session()

    def _request(self, method: str, path: str, **kwargs):
        delay, error = 0.5, None
        for attempt in range(self.retries):
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

    def set_routing_metric(self, metric: str) -> None:
        """Floodlight's own routing: 'hopcount' (default) or 'latency' (QoS-aware)."""
        self._request("POST", "/wm/routing/metric/json", json={"metric": metric})

    def push_flow(self, flow: dict) -> None:
        result = self._request("POST", "/wm/staticentrypusher/json", json=flow)
        if "pushed" not in str(result.get("status", "")).lower():
            raise ControllerError(f"Flow {flow.get('name')} rejected: {result}")

    def delete_flow(self, name: str) -> None:
        self._request("DELETE", "/wm/staticentrypusher/json", data=json.dumps({"name": name}))

    def clear_flows(self) -> None:
        """Remove static entries from every connected switch."""
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


# ----------------------------------------------------------------------------------------
# Flow rules
# ----------------------------------------------------------------------------------------
class FlowInstaller:
    """Pushes / removes bidirectional static entries for a switch path between two hosts."""

    def __init__(self, client: FloodlightClient, spec):
        self.client, self.spec = client, spec

    def install_path(self, src_ip: str, dst_ip: str, path: list[str], prefix: str,
                     priority: int = 32768, ip_protos=None) -> list[str]:
        src, dst = self.spec.host_by_ip(src_ip), self.spec.host_by_ip(dst_ip)
        if path[0] != src.switch or path[-1] != dst.switch:
            raise ValueError(f"Path {path} does not connect {src.name} to {dst.name}")
        names = []
        try:
            for i, switch in enumerate(path):
                in_port = src.switch_port if i == 0 else self.spec.port(switch, path[i - 1])
                out_port = dst.switch_port if i == len(path) - 1 else self.spec.port(switch, path[i + 1])
                for proto in (ip_protos or [None]):
                    for direction, (a, b, port_in, port_out) in enumerate(
                            ((src_ip, dst_ip, in_port, out_port), (dst_ip, src_ip, out_port, in_port))):
                        name = f"{prefix}_{switch}_{direction}" + ("" if proto is None else f"_p{proto}")
                        flow = {"switch": self.spec.switch_dpids[switch], "name": name, "priority": str(priority),
                                "eth_type": "0x0800", "ipv4_src": a, "ipv4_dst": b, "in_port": str(port_in),
                                "active": "true", "actions": f"output={port_out}"}
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
# Path statistics (the 19 features)
# ----------------------------------------------------------------------------------------
class PathProber:
    """
    Measures the installed path like the AIRSDN data collector:
      ping -f -c 500 client -> server                 -> average_rtt, packet_loss
      iperf3 -R -t 30 (server sends to the client)    -> bits_per_second, bu_ratio, retransmits, cpu
      NFStream on the client's access switch port     -> 12 duration / inter-arrival features of the
                                                         iperf3 TCP flow (idle_timeout 15 s, as in the paper)
    """

    def __init__(self, net, spec, iperf_port=5201, probe_seconds=30, ping_count=500, idle_timeout=15):
        self.net, self.spec = net, spec
        self.iperf_port, self.probe_seconds = iperf_port, probe_seconds
        self.ping_count, self.idle_timeout = ping_count, idle_timeout

    def measure(self, client_host: str, server_host: str) -> list[float]:
        import pingparsing
        from nfstream import NFStreamer

        client, server = self.net.get(client_host), self.net.get(server_host)
        client_ip, server_ip = self.spec.hosts[client_host].ip, self.spec.hosts[server_host].ip

        parsed = pingparsing.PingParsing().parse(client.cmd(f"ping {server_ip} -c {self.ping_count} -f")).as_dict()
        avg_rtt = parsed.get("rtt_avg")
        loss = parsed.get("packet_loss_rate")
        avg_rtt = 10000.0 if avg_rtt is None else float(avg_rtt)    # unreachable = maximally congested
        loss = 100.0 if loss is None else float(loss)

        captured, stop = [], threading.Event()

        def listen():
            try:
                for flow in NFStreamer(source=self.spec.access_interface(client_host),
                                       statistical_analysis=True, idle_timeout=self.idle_timeout):
                    if ({flow.src_ip, flow.dst_ip} == {client_ip, server_ip} and flow.protocol == 6
                            and self.iperf_port in (flow.src_port, flow.dst_port)):
                        captured.append(flow)
                        break
                    if stop.is_set():
                        break
            except Exception as exc:
                log.warning("NFStream capture failed: %s", exc)

        listener = threading.Thread(target=listen, daemon=True)
        listener.start()
        time.sleep(1)
        iperf_server = server.popen(["iperf3", "-s", "-1", "-p", str(self.iperf_port)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)
        iperf = client.popen(["iperf3", "-c", server_ip, "-p", str(self.iperf_port), "-b", "0", "-R",
                              "-t", str(self.probe_seconds), "-J"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        try:
            stdout, _ = iperf.communicate(timeout=self.probe_seconds + 60)
        except subprocess.TimeoutExpired:
            iperf.kill()
            stdout = ""
        if iperf_server.poll() is None:
            iperf_server.terminate()
        stop.set()
        client.cmd(f"ping {server_ip} -c 5 -f")      # wakes the NFStream loop (as the original code does)
        listener.join(timeout=self.idle_timeout + 30)

        try:
            end = json.loads(stdout)["end"]
        except (ValueError, KeyError, TypeError):
            log.warning("iperf3 produced no result on this path")
            end = {}
        received = end.get("sum_received") or end.get("sum") or {}
        sent = end.get("sum_sent") or end.get("sum") or {}
        cpu = end.get("cpu_utilization_percent") or {}
        bps = float(received.get("bits_per_second", 0.0))
        features = [avg_rtt, loss, bps, 100 - bps / (self.spec.link_bw_mbps * 1e6) * 100,
                    int(sent.get("retransmits", 0)), float(cpu.get("host_total", 0.0)),
                    float(cpu.get("host_system", 0.0))]
        if captured:
            features += [float(getattr(captured[0], name)) for name in NFSTREAM_FEATURES]
        else:
            log.warning("NFStream captured no iperf3 flow; using zeros (as in the training data)")
            features += [0.0] * len(NFSTREAM_FEATURES)
        return features


# ----------------------------------------------------------------------------------------
# Logistic Regression
# ----------------------------------------------------------------------------------------
class TrafficClassifier:
    """The trained Logistic Regression model (models/logistic_regression_model.pkl)."""

    def __init__(self, model_path: str):
        import joblib
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path} (run airsdn_reproduction/train_lr.py first)")
        self.model = joblib.load(model_path)
        estimator = self.model.steps[-1][1] if hasattr(self.model, "steps") else self.model
        if type(estimator).__name__ != "LogisticRegression":
            raise TypeError(f"Expected a Logistic Regression model, got {type(estimator).__name__}")
        if getattr(self.model, "n_features_in_", len(FEATURE_NAMES)) != len(FEATURE_NAMES):
            raise ValueError("Model does not take the 19 AIRSDN features")
        self.high_index = list(self.model.classes_).index(max(self.model.classes_))

    def predict(self, features: list[float]) -> tuple[int, float, float]:
        """Returns (traffic level: 1 = low, 2 = high; P(high); inference time in ms)."""
        row = np.asarray([features], dtype=float)
        if not np.all(np.isfinite(row)):
            raise ValueError(f"Non-finite feature vector: {features}")
        start = time.perf_counter()
        level = int(self.model.predict(row)[0])
        probability = float(self.model.predict_proba(row)[0][self.high_index])
        return level, probability, (time.perf_counter() - start) * 1000


@dataclass
class ScoredPath:
    index: int
    switches: list[str]
    features: dict
    traffic_level: int
    p_high_traffic: float
    probe_s: float
    inference_ms: float


@dataclass
class SelectionResult:
    chosen_index: int
    chosen: list[str]
    scored: list[ScoredPath] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def routing_algorithm(candidates: list[list[str]], client_host: str, server_host: str,
                      installer: FlowInstaller, prober: PathProber, classifier: TrafficClassifier,
                      route_priority: int = 32768) -> tuple[SelectionResult, list[str]]:
    """
    The AIRSDN routing algorithm (paper Table 2). Returns the selection and the names of the
    entries installed for the chosen path.
    """
    if not candidates:
        raise ValueError("No candidate paths")
    spec = prober.spec
    client_ip, server_ip = spec.hosts[client_host].ip, spec.hosts[server_host].ip
    scored = []
    for index, path in enumerate(candidates):
        names = installer.install_path(client_ip, server_ip, path, prefix=f"probe{index}",
                                       priority=40000, ip_protos=[1, 6])
        started = time.monotonic()
        try:
            features = prober.measure(client_host, server_host)
        finally:
            installer.remove(names)
        level, probability, inference_ms = classifier.predict(features)
        scored.append(ScoredPath(index, path, dict(zip(FEATURE_NAMES, features)), level, round(probability, 6),
                                 round(time.monotonic() - started, 2), round(inference_ms, 4)))
        log.info("Path %d %s: traffic level %d (P(high)=%.3f, rtt=%.0f ms, %.2f Mbps)", index, "-".join(path),
                 level, probability, features[0], features[2] / 1e6)
        time.sleep(1)
    best = min(scored, key=lambda s: (s.traffic_level, s.index))     # lowest level, first on ties
    installed = installer.install_path(client_ip, server_ip, best.switches, prefix="airsdn", priority=route_priority)
    return SelectionResult(best.index, best.switches, scored), installed
