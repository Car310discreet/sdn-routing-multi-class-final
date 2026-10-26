"""
path_discovery.py - Stage 1 of AIRSDN: hop-count candidate paths.

Yen's K-shortest loopless paths with every link weighted 1 (hop count), K = 5 as in the paper
("the five shortest paths between the client and server are taken from the Floodlight
controller"). Ties are broken by graph insertion order, which on NSFNET reproduces Floodlight's
own ranking: s0 -> s9 gives 0-3-8-9, 0-1-7-10-9, 0-2-5-12-9, 0-3-4-5-12-9, 0-3-8-11-10-9.

    python airsdn_reproduction/path_discovery.py            # print the candidates for the paper's scenario
"""
from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import asdict, dataclass, field

import networkx as nx

K_PATHS = 5


@dataclass
class DiscoveryResult:
    src: str
    dst: str
    k: int
    compute_ms: float
    paths: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _shortest_path(graph: nx.Graph, src: str, dst: str, banned_nodes, banned_edges):
    """Breadth-first search (unit weights) avoiding the banned nodes/edges; returns a path or None."""
    if src in banned_nodes or dst in banned_nodes:
        return None
    parent = {src: None}
    frontier = [src]
    while frontier:
        nxt = []
        for node in frontier:
            for neighbour in graph.neighbors(node):
                if neighbour in parent or neighbour in banned_nodes or (node, neighbour) in banned_edges:
                    continue
                parent[neighbour] = node
                if neighbour == dst:
                    path = [dst]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    return path[::-1]
                nxt.append(neighbour)
        frontier = nxt
    return None


def yen_hop_count(graph: nx.Graph, src: str, dst: str, k: int = K_PATHS) -> list[list[str]]:
    """Yen's algorithm with hop count as the path cost; up to k loopless paths, shortest first."""
    first = _shortest_path(graph, src, dst, frozenset(), frozenset())
    if first is None:
        return []
    accepted, seen = [first], {tuple(first)}
    candidates, counter = [], itertools.count()
    while len(accepted) < k:
        last = accepted[-1]
        for i in range(len(last) - 1):
            root = last[:i + 1]
            banned_edges = set()
            for path in accepted:
                if path[:i + 1] == root:
                    banned_edges |= {(path[i], path[i + 1]), (path[i + 1], path[i])}
            spur = _shortest_path(graph, last[i], dst, frozenset(root[:-1]), banned_edges)
            if spur is None:
                continue
            total = root[:-1] + spur
            if tuple(total) not in seen:
                seen.add(tuple(total))
                heapq.heappush(candidates, (len(total) - 1, next(counter), total))
        if not candidates:
            break
        accepted.append(heapq.heappop(candidates)[2])
    return accepted


def discover_paths(graph: nx.Graph, src: str, dst: str, k: int = K_PATHS) -> DiscoveryResult:
    """Stage 1: the k hop-count shortest switch paths from src to dst, with compute time."""
    if src not in graph or dst not in graph:
        raise ValueError(f"Unknown switch: {src if src not in graph else dst}")
    if k < 1:
        raise ValueError("k must be >= 1")
    start = time.perf_counter()
    paths = yen_hop_count(graph, src, dst, k)
    return DiscoveryResult(src, dst, k, round((time.perf_counter() - start) * 1000, 4), paths)


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from topology_builder import CLIENT, SERVER, build_nsfnet

    spec = build_nsfnet()
    result = discover_paths(spec.graph, spec.hosts[CLIENT].switch, spec.hosts[SERVER].switch)
    print(f"{len(result.paths)} hop-count candidates in {result.compute_ms} ms:")
    for i, path in enumerate(result.paths):
        print(f"  {i}: {'-'.join(path)} ({len(path) - 1} hops)")
