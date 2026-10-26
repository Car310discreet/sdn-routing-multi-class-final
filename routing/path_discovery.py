"""
path_discovery.py - Module 2, Stage 1: multi-metric candidate path discovery.

Four togglable methods return the top-K candidate switch paths (K = 5-10):

  hop        Yen's K-shortest paths, link weight = 1 (the AIRSDN / GEANT2 baseline)
  latency    Yen's K-shortest paths, link weight = measured delay (ms)
  bandwidth  K widest paths: Yen's framework over a max-bottleneck Dijkstra with a max-priority
             queue; ranked by bottleneck available bandwidth, then hops
  composite  Yen's K-shortest paths over the normalised composite link cost
               cost = alpha*D/Dmax + beta*J/Jmax + gamma/max(B/Bmax, 1e-3) + delta*L
             D delay, J jitter, B available bandwidth, L loss rate (0-1); each metric is normalised
             by its maximum over all links, so the weights are unit-free

Every candidate also carries its composite cost, whatever the method, so Stage 2 can always break
ties on it. Every search is hop-bounded (shortest hop distance + `hop_slack`, exact: the search runs
over (node, hops) states), and every discovery reports its own computation time in milliseconds.

Link metrics are read from switch-graph edge attributes: delay_ms, jitter_ms, avail_bw_mbps, loss_rate
(telemetry_and_model.LinkTelemetry writes them; configured delay / full capacity when missing).

Offline, on GEANT2 with synthetic link metrics (no root):
    python routing/path_discovery.py --k 5
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import networkx as nx

METHODS = ("hop", "latency", "bandwidth", "composite")
EPSILON = 1e-3          # floor for normalised available bandwidth (avoids 1/0 on saturated links)
DEFAULT_HOP_SLACK = 4
DEFAULT_WEIGHTS = {"alpha": 0.3, "beta": 0.1, "gamma": 0.4, "delta": 0.2}


@dataclass
class CandidatePath:
    switches: list[str]
    cost: float                 # method-specific: hops, ms, -bottleneck Mbit/s, or composite cost
    composite_cost: float       # composite cost of the path, for every method (Stage-2 tie-break)
    hops: int
    delay_ms: float
    bottleneck_mbps: float


@dataclass
class DiscoveryResult:
    method: str
    src: str
    dst: str
    k: int
    max_hops: int
    compute_ms: float
    weights: dict
    paths: list[CandidatePath] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------------------
# Link metrics
# ----------------------------------------------------------------------------------------
def _edge_value(data: dict, key: str, default: float) -> float:
    value = data.get(key)
    return default if value is None else float(value)


def link_delay(data: dict) -> float:
    return max(0.0, _edge_value(data, "delay_ms", 0.0))


def link_avail_bw(data: dict) -> float:
    return max(0.0, _edge_value(data, "avail_bw_mbps", _edge_value(data, "capacity_mbps", 0.0)))


def composite_link_costs(graph: nx.Graph, alpha=0.3, beta=0.1, gamma=0.4, delta=0.2) -> dict:
    """Normalised composite cost of every link, keyed by both directions."""
    if min(alpha, beta, gamma, delta) < 0:
        raise ValueError("Composite weights must be non-negative")
    edges = list(graph.edges(data=True))
    if not edges:
        return {}
    d_max = max(link_delay(d) for _, _, d in edges) or 1.0
    j_max = max(_edge_value(d, "jitter_ms", 0.0) for _, _, d in edges) or 1.0
    b_max = max(link_avail_bw(d) for _, _, d in edges) or 1.0
    costs = {}
    for u, v, d in edges:
        norm_bw = max(link_avail_bw(d) / b_max, EPSILON)
        loss = min(max(_edge_value(d, "loss_rate", 0.0), 0.0), 1.0)
        cost = (alpha * link_delay(d) / d_max + beta * _edge_value(d, "jitter_ms", 0.0) / j_max
                + gamma / norm_bw + delta * loss)
        costs[(u, v)] = costs[(v, u)] = cost
    return costs


# ----------------------------------------------------------------------------------------
# Hop-limited searches
# ----------------------------------------------------------------------------------------
def shortest_path_bounded(graph: nx.Graph, src: str, dst: str, weight: Callable[[str, str], float],
                          max_hops: int, banned_nodes=frozenset(), banned_edges=frozenset()):
    """Exact minimum-weight path with at most `max_hops` links (Dijkstra over (node, hops) states)."""
    if src in banned_nodes or dst in banned_nodes:
        return None
    best = {(src, 0): 0.0}
    parent = {(src, 0): None}
    counter = itertools.count()
    heap = [(0.0, next(counter), src, 0)]
    while heap:
        cost, _, node, hops = heapq.heappop(heap)
        if cost > best.get((node, hops), float("inf")):
            continue
        if node == dst:
            path, state = [], (node, hops)
            while state is not None:
                path.append(state[0])
                state = parent[state]
            return cost, path[::-1]
        if hops == max_hops:
            continue
        for nxt in graph.neighbors(node):
            if nxt in banned_nodes or (node, nxt) in banned_edges:
                continue
            new_cost = cost + weight(node, nxt)
            state = (nxt, hops + 1)
            if new_cost < best.get(state, float("inf")):
                best[state] = new_cost
                parent[state] = (node, hops)
                heapq.heappush(heap, (new_cost, next(counter), nxt, hops + 1))
    return None


def widest_path_bounded(graph: nx.Graph, src: str, dst: str, bandwidth: Callable[[str, str], float],
                        max_hops: int, banned_nodes=frozenset(), banned_edges=frozenset()):
    """Modified Dijkstra with a max-priority queue on bottleneck bandwidth; ties to fewer hops."""
    if src in banned_nodes or dst in banned_nodes:
        return None
    best = {(src, 0): float("inf")}
    parent = {(src, 0): None}
    counter = itertools.count()
    heap = [(-float("inf"), 0, next(counter), src)]   # max-heap via negated bottleneck
    while heap:
        neg_width, hops, _, node = heapq.heappop(heap)
        width = -neg_width
        if width < best.get((node, hops), -1.0):
            continue
        if node == dst:
            path, state = [], (node, hops)
            while state is not None:
                path.append(state[0])
                state = parent[state]
            return (-width, hops), path[::-1]
        if hops == max_hops:
            continue
        for nxt in graph.neighbors(node):
            if nxt in banned_nodes or (node, nxt) in banned_edges:
                continue
            new_width = min(width, bandwidth(node, nxt))
            state = (nxt, hops + 1)
            if new_width > best.get(state, -1.0):
                best[state] = new_width
                parent[state] = (node, hops)
                heapq.heappush(heap, (-new_width, hops + 1, next(counter), nxt))
    return None


def yen_k_paths(graph: nx.Graph, src: str, dst: str, k: int, max_hops: int, search: Callable,
                path_key: Callable[[list[str]], tuple | float]) -> list[tuple]:
    """Yen's algorithm over any base search; returns up to k (key, path) tuples, best first."""
    first = search(src, dst, max_hops, frozenset(), frozenset())
    if first is None:
        return []
    accepted = [(path_key(first[1]), first[1])]
    seen = {tuple(first[1])}
    candidates, counter = [], itertools.count()
    while len(accepted) < k:
        last_path = accepted[-1][1]
        for i in range(len(last_path) - 1):
            spur_node, root = last_path[i], last_path[:i + 1]
            banned_edges = set()
            for _, path in accepted:
                if path[:i + 1] == root:
                    banned_edges.update({(path[i], path[i + 1]), (path[i + 1], path[i])})
            remaining_hops = max_hops - i
            if remaining_hops < 1:
                continue
            spur = search(spur_node, dst, remaining_hops, frozenset(root[:-1]), frozenset(banned_edges))
            if spur is None:
                continue
            total = root[:-1] + spur[1]
            if tuple(total) not in seen:
                seen.add(tuple(total))
                heapq.heappush(candidates, (path_key(total), next(counter), total))
        if not candidates:
            break
        key, _, path = heapq.heappop(candidates)
        accepted.append((key, path))
    return accepted


# ----------------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------------
def discover_paths(graph: nx.Graph, src: str, dst: str, method: str, k: int = 5,
                   max_hops: int | None = None, hop_slack: int = DEFAULT_HOP_SLACK,
                   weights: dict | None = None) -> DiscoveryResult:
    """Run one Stage-1 method; return its top-k candidates and its compute time."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    if src not in graph or dst not in graph:
        raise ValueError(f"Unknown switch: {src if src not in graph else dst}")
    if k < 1:
        raise ValueError("k must be >= 1")
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    start = time.perf_counter()
    costs = composite_link_costs(graph, **weights)
    if src == dst:
        paths, bound = [(0.0, [src])], 0
    else:
        try:
            bound = max_hops if max_hops is not None else nx.shortest_path_length(graph, src, dst) + hop_slack
        except nx.NetworkXNoPath:
            raise ValueError(f"No path between {src} and {dst}") from None
        if method == "bandwidth":
            bw = lambda u, v: link_avail_bw(graph.edges[u, v])  # noqa: E731
            search = lambda s, t, h, bn, be: widest_path_bounded(graph, s, t, bw, h, bn, be)  # noqa: E731

            def path_key(path):
                return (-min(bw(u, v) for u, v in zip(path, path[1:])), len(path) - 1)
        else:
            if method == "hop":
                weight = lambda u, v: 1.0  # noqa: E731
            elif method == "latency":
                weight = lambda u, v: link_delay(graph.edges[u, v])  # noqa: E731
            else:
                weight = lambda u, v: costs[(u, v)]  # noqa: E731
            search = lambda s, t, h, bn, be: shortest_path_bounded(graph, s, t, weight, h, bn, be)  # noqa: E731
            path_key = lambda path: sum(weight(u, v) for u, v in zip(path, path[1:]))  # noqa: E731
        paths = yen_k_paths(graph, src, dst, k, bound, search, path_key)
    described = []
    for key, path in paths:
        edges = list(zip(path, path[1:]))
        described.append(CandidatePath(
            switches=path, cost=round(float(key[0] if isinstance(key, tuple) else key), 6),
            composite_cost=round(sum(costs[e] for e in edges), 6), hops=len(edges),
            delay_ms=round(sum(link_delay(graph.edges[e]) for e in edges), 3),
            bottleneck_mbps=round(min((link_avail_bw(graph.edges[e]) for e in edges), default=0.0), 3)))
    compute_ms = (time.perf_counter() - start) * 1000
    return DiscoveryResult(method, src, dst, k, bound, round(compute_ms, 4), weights, described)


def synthetic_link_metrics(graph: nx.Graph, seed: int = 1) -> nx.Graph:
    """Copy of `graph` with plausible random telemetry (offline tests and benchmarks only)."""
    rng = random.Random(seed)
    graph = graph.copy()
    for _, _, d in graph.edges(data=True):
        capacity = d.get("capacity_mbps", 10.0)
        d["delay_ms"] = round(d.get("delay_ms", 10.0) + rng.uniform(0, 30), 1)   # propagation + queueing
        d["jitter_ms"] = round(rng.uniform(0, 3), 2)
        d["avail_bw_mbps"] = round(capacity * rng.uniform(0.0, 1.0), 2)
        d["loss_rate"] = round(rng.choice([0, 0, 0, rng.uniform(0, 0.1)]), 4)
    return graph


def main():
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from topology_builder import RECEIVER, SENDER, build_geant2

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--hop-slack", type=int, default=DEFAULT_HOP_SLACK)
    parser.add_argument("--seed", type=int, default=1, help="synthetic telemetry seed")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    spec = build_geant2()
    graph = synthetic_link_metrics(spec.graph, args.seed)
    src, dst = spec.hosts[SENDER].switch, spec.hosts[RECEIVER].switch
    results = {m: discover_paths(graph, src, dst, m, args.k, hop_slack=args.hop_slack) for m in METHODS}
    if args.json:
        print(json.dumps({m: r.to_dict() for m, r in results.items()}, indent=2))
        return
    for method, result in results.items():
        print(f"\n[{method}] {len(result.paths)} paths in {result.compute_ms:.3f} ms (max hops {result.max_hops})")
        for p in result.paths:
            print(f"  cost={p.cost:<10g} composite={p.composite_cost:<9g} hops={p.hops} delay={p.delay_ms}ms "
                  f"bottleneck={p.bottleneck_mbps}Mbps  {'-'.join(p.switches)}")


if __name__ == "__main__":
    main()
