"""
topology_builder.py - Module 1: the evaluation networks (GEANT2, k=4 fat-tree) and their 4-class traffic.

Topologies (Mininet + NetworkX), `build_topology(name)`:
  geant2   GEANT2 as used by RouteNet / DRL-GNN: 24 switches, 37 links (edge list from
           knowledgedefinednetworking/DRL-GNN).
               Sender h15 (s15) -> Receiver h23 (s23), 6 hops apart (the network diameter)
  fattree  k=4 fat-tree data-centre fabric: 20 switches, 32 links. Core s0-s3; pod p has aggregation
           switches s(4+2p), s(5+2p) and edge switches s(12+2p), s(13+2p).
               Sender h12 (s12, pod 0) -> Receiver h18 (s18, pod 3), 4 hops apart; 4 equal-cost shortest paths
Both use the conditions their datasets were collected under, so Stage 2 features stay in the models'
training distribution: switch-switch links 10 Mbit/s, 10 ms, a 1,000-packet queue; one host per switch
(h<i> on s<i>, 10.0.0.<i+1>; on the fat-tree, hosts on core / aggregation switches only source and sink
background traffic); access links not shaped.

Startup (`start_geant2`)
  IPv6 is disabled on every interface while the network is built: with it, Open vSwitch reports a
  pkt_mark field in PacketIns that Floodlight cannot parse, so Floodlight drops switches and can keep
  stale routes. Hosts get static ARP. pingAll must succeed (a few isolated timeouts allowed) and the
  Sender <-> Receiver pair must answer.

Traffic (`TrafficGenerator`, `Scenario`)
  Every link carries one iperf3 UDP flow per direction between the hosts at its ends, pinned onto that
  link with static entries, at a rate set by the link's class:
      0 Extremely Low [0.00, 0.25)   1 Low [0.25, 0.50)   2 High [0.50, 0.80)   3 Extremely High [0.80, 1.20]
  (fraction of capacity). A path's ground-truth class is the class of its most loaded link. Link classes
  are drawn so that path classes are balanced for the topology's Sender -> Receiver path length h
  (6 on GEANT2, 4 on the fat-tree): P(link class <= c) = ((c + 1) / 4) ** (1 / h).

    python routing/topology_builder.py [--topology fattree]             # offline summary
    python routing/topology_builder.py [--topology fattree] --preview   # class balance of sampled scenarios
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from mininet.topo import Topo

# Importing Mininet makes every later logger a level-25 MininetLogger, which hides INFO messages
logging.setLoggerClass(logging.Logger)
log = logging.getLogger(__name__)

NUM_SWITCHES = 24          # GEANT2
GEANT2_LINKS = [
    (0, 1), (0, 2), (1, 3), (1, 6), (1, 9), (2, 3), (2, 4), (3, 6), (4, 7), (5, 3),
    (5, 8), (6, 9), (6, 8), (7, 11), (7, 8), (8, 11), (8, 20), (8, 17), (8, 18), (8, 12),
    (9, 10), (9, 13), (9, 12), (10, 13), (11, 20), (11, 14), (12, 13), (12, 19), (12, 21),
    (14, 15), (15, 16), (16, 17), (17, 18), (18, 21), (19, 23), (21, 22), (22, 23),
]


def fat_tree_links(k: int = 4) -> list[tuple[int, int]]:
    """Core switches first, then aggregation, then edge; aggregation switch a of a pod -> core a*k/2 ...;
    every edge switch -> both aggregation switches of its pod."""
    half = k // 2
    agg, edge = half * half, half * half + k * half
    links = []
    for pod in range(k):
        for a in range(half):
            links += [(c, agg + pod * half + a) for c in range(a * half, (a + 1) * half)]
        for e in range(half):
            links += [(agg + pod * half + a, edge + pod * half + e) for a in range(half)]
    return links


FATTREE_LINKS = fat_tree_links(4)
FATTREE_EDGE_SWITCHES = [f"s{i}" for i in range(12, 20)]
TOPOLOGIES = {
    # links, switches, Sender, Receiver, path length the scenario class balance targets
    "geant2": dict(links=GEANT2_LINKS, num_switches=NUM_SWITCHES, sender="h15", receiver="h23", path_hops=6.0),
    "fattree": dict(links=FATTREE_LINKS, num_switches=20, sender="h12", receiver="h18", path_hops=4.0),
}
LINK_BW_MBPS = 10.0
LINK_DELAY_MS = 10.0
QUEUE_SIZE = 1000
SENDER, RECEIVER = "h15", "h23"          # GEANT2; every spec carries its own (spec.sender, spec.receiver)

CLASS_NAMES = ["Extremely Low", "Low", "High", "Extremely High"]
CLASS_BANDS = [(0.00, 0.25), (0.25, 0.50), (0.50, 0.80), (0.80, 1.20)]


# ----------------------------------------------------------------------------------------
# Topology
# ----------------------------------------------------------------------------------------
@dataclass
class HostInfo:
    name: str
    ip: str
    switch: str
    switch_port: int


@dataclass
class TopologySpec:
    topo: Topo
    graph: nx.Graph                      # switch graph; edges: ports, delay_ms, capacity_mbps
    hosts: dict[str, HostInfo]
    switch_dpids: dict[str, str]         # switch -> Floodlight DPID (colon form)
    link_bw_mbps: float = LINK_BW_MBPS
    name: str = "geant2"
    sender: str = SENDER
    receiver: str = RECEIVER
    path_hops: float = 6.0               # path length the scenario class balance targets

    @property
    def dpid_to_switch(self) -> dict[str, str]:
        return {d: s for s, d in self.switch_dpids.items()}

    def port(self, u: str, v: str) -> int:
        """Port on switch u of the link to switch v."""
        return self.graph.edges[u, v]["ports"][u]

    def host_by_ip(self, ip: str) -> HostInfo:
        for info in self.hosts.values():
            if info.ip == ip:
                return info
        raise KeyError(f"No host with IP {ip}")

    def access_interface(self, host: str) -> str:
        """Root-namespace interface of the switch port facing `host` (e.g. s23-eth3)."""
        info = self.hosts[host]
        return f"{info.switch}-eth{info.switch_port}"

    def hosts_on(self, switch: str) -> list[str]:
        return sorted((h for h, i in self.hosts.items() if i.switch == switch), key=lambda h: (len(h), h))


def dpid(index: int) -> str:
    return f"{index + 1:016x}"


class SwitchGraphTopo(Topo):
    def build(self, links: list, num_switches: int, delays: dict, bw: float = LINK_BW_MBPS, queue: int = QUEUE_SIZE):
        for i in range(num_switches):
            self.addSwitch(f"s{i}", dpid=dpid(i), protocols="OpenFlow13")
        for a, b in links:
            self.addLink(f"s{a}", f"s{b}", bw=bw, delay=f"{delays[(a, b)]}ms", max_queue_size=queue, use_htb=True)
        for i in range(num_switches):
            host = self.addHost(f"h{i}", ip=f"10.0.0.{i + 1}/8")
            self.addLink(f"s{i}", host)          # access link: not shaped


def build_topology(name: str = "geant2", link_bw_mbps: float = LINK_BW_MBPS) -> TopologySpec:
    if name not in TOPOLOGIES:
        raise ValueError(f"Unknown topology {name!r}; choose from {sorted(TOPOLOGIES)}")
    config = TOPOLOGIES[name]
    links, num_switches = config["links"], config["num_switches"]
    delays = {link: LINK_DELAY_MS for link in links}
    topo = SwitchGraphTopo(links=links, num_switches=num_switches, delays=delays, bw=link_bw_mbps)
    graph = nx.Graph()
    graph.add_nodes_from(f"s{i}" for i in range(num_switches))
    for a, b in links:
        u, v = f"s{a}", f"s{b}"
        port_u, port_v = topo.port(u, v)
        graph.add_edge(u, v, ports={u: port_u, v: port_v}, delay_ms=delays[(a, b)], capacity_mbps=link_bw_mbps)
    hosts = {}
    for i in range(num_switches):
        switch_port, _ = topo.port(f"s{i}", f"h{i}")
        hosts[f"h{i}"] = HostInfo(f"h{i}", f"10.0.0.{i + 1}", f"s{i}", switch_port)
    dpids = {f"s{i}": ":".join(dpid(i)[k:k + 2] for k in range(0, 16, 2)) for i in range(num_switches)}
    return TopologySpec(topo, graph, hosts, dpids, link_bw_mbps, name, config["sender"], config["receiver"],
                        config["path_hops"])


def build_geant2(link_bw_mbps: float = LINK_BW_MBPS) -> TopologySpec:
    return build_topology("geant2", link_bw_mbps)


def build_fattree(link_bw_mbps: float = LINK_BW_MBPS) -> TopologySpec:
    return build_topology("fattree", link_bw_mbps)


# ----------------------------------------------------------------------------------------
# Network lifecycle (root)
# ----------------------------------------------------------------------------------------
IPV6_DEFAULT = "/proc/sys/net/ipv6/conf/default/disable_ipv6"


def start_topology(spec: TopologySpec, api, ping_all: bool = True, settle_s: float = 30.0,
                   max_loss_pct: float = 1.0, controller_ip="127.0.0.1", controller_port=6653):
    """Start the network on Floodlight with IPv6 off and static ARP; check connectivity. Needs root."""
    from mininet.link import TCLink
    from mininet.net import Mininet
    from mininet.node import OVSKernelSwitch, RemoteController
    from telemetry_and_model import ControllerError, wait_for_switches

    with open(IPV6_DEFAULT) as handle:
        previous = handle.read().strip()
    with open(IPV6_DEFAULT, "w") as handle:        # only interfaces created while this is set
        handle.write("1")
    net = None
    try:
        net = Mininet(topo=spec.topo, switch=OVSKernelSwitch, link=TCLink, controller=None,
                      autoSetMacs=True, build=False, waitConnected=True)
        net.addController("c1", controller=RemoteController, ip=controller_ip, port=controller_port,
                          protocols="OpenFlow13")
        net.build()
        net.start()
    except BaseException:
        if net is not None:
            net.stop()
        raise
    finally:
        with open(IPV6_DEFAULT, "w") as handle:
            handle.write(previous)
    try:
        for host in net.hosts:
            host.cmd("sysctl -qw net.ipv6.conf.all.disable_ipv6=1")
        net.staticArp()
        wait_for_switches(api, set(spec.switch_dpids.values()))
        api.clear_flows()            # static entries of earlier runs reappear on reused DPIDs
        api.set_routing_metric("hopcount")
        if ping_all:
            endpoints = [net.get(spec.sender), net.get(spec.receiver)]
            for attempt in range(2):
                loss = net.pingAll()
                pair_loss = net.ping(endpoints, timeout="2")
                if loss <= max_loss_pct and pair_loss == 0:
                    break
                if attempt == 0:
                    log.warning("pingAll lost %.1f %% (Sender <-> Receiver %.0f %%); waiting %.0f s", loss,
                                pair_loss, settle_s)
                    time.sleep(settle_s)
            else:
                raise ControllerError(f"Hosts cannot reach each other through Floodlight ({loss:.1f} % loss)")
    except BaseException:
        stop_network(net)
        raise
    return net


start_geant2 = start_topology


def stop_network(net) -> None:
    if net is None:
        return
    for host in net.hosts:
        if host.waiting:
            try:
                host.sendInt()
                host.waitOutput()
            except Exception as exc:
                log.warning("Could not interrupt %s: %s", host.name, exc)
    net.stop()


def chown_to_invoking_user(path: str) -> None:
    """Outputs are written as root under sudo; hand them back to the user who ran sudo."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid is None or gid is None or not os.path.exists(path):
        return
    for root, dirs, files in os.walk(path):
        for name in [root] + [os.path.join(root, n) for n in dirs + files]:
            try:
                os.chown(name, int(uid), int(gid))
            except OSError:
                pass


# ----------------------------------------------------------------------------------------
# 4-class traffic
# ----------------------------------------------------------------------------------------
def class_of(utilization: float) -> int:
    for index, (_, high) in enumerate(CLASS_BANDS[:-1]):
        if utilization < high:
            return index
    return len(CLASS_BANDS) - 1


def link_key(u: str, v: str) -> tuple[str, str]:
    return (u, v) if int(u[1:]) < int(v[1:]) else (v, u)


def link_class_weights(path_hops: float = 6.0) -> np.ndarray:
    """P(link class = c) such that the max class over `path_hops` links is uniform over the classes."""
    cumulative = np.array([((c + 1) / len(CLASS_NAMES)) ** (1.0 / path_hops) for c in range(len(CLASS_NAMES))])
    return np.diff(np.concatenate([[0.0], cumulative]))


@dataclass
class Scenario:
    utilization: dict[tuple[str, str], float]      # link_key -> offered load / capacity, each direction
    seed: int | None = None

    def link(self, u: str, v: str) -> float:
        return self.utilization[link_key(u, v)]

    def path_utilization(self, path: list[str]) -> float:
        return max(self.link(u, v) for u, v in zip(path, path[1:]))

    def path_class(self, path: list[str]) -> int:
        return class_of(self.path_utilization(path))

    def congest(self, path: list[str], utilization: float = 1.0) -> None:
        """Put every link of `path` in Extremely High (the AIRSDN scenario: congested default path)."""
        for u, v in zip(path, path[1:]):
            self.utilization[link_key(u, v)] = utilization

    def to_dict(self) -> dict:
        return {"seed": self.seed, "utilization": {f"{u}-{v}": round(x, 4) for (u, v), x in self.utilization.items()}}


def targeted_weights(target_class: int, path_hops: float, target_share: float = 0.3) -> np.ndarray:
    """Link-class weights with no class above `target_class` and `target_share` of the links in it, so that most
    multi-hop paths get a `target_class` bottleneck (the dataset collector fills lagging classes with these)."""
    weights = link_class_weights(path_hops)
    weights[target_class + 1:] = 0.0
    if target_class == 0:
        return weights / weights.sum()
    lower = weights[:target_class] / weights[:target_class].sum() * (1.0 - target_share)
    weights[:target_class], weights[target_class] = lower, target_share
    return weights


def sample_scenario(graph: nx.Graph, seed: int | None = None, path_hops: float = 6.0,
                    target_class: int | None = None) -> Scenario:
    rng = np.random.default_rng(seed)
    weights = link_class_weights(path_hops) if target_class is None else targeted_weights(target_class, path_hops)
    utilization = {}
    for u, v in sorted(link_key(a, b) for a, b in graph.edges):
        low, high = CLASS_BANDS[rng.choice(len(CLASS_BANDS), p=weights)]
        utilization[(u, v)] = float(rng.uniform(low, high))
    return Scenario(utilization, seed)


@dataclass
class TrafficGenerator:
    """Pinned per-link iperf3 UDP flows whose rates follow the current Scenario (needs Mininet)."""
    net: object
    spec: TopologySpec
    installer: object                          # telemetry_and_model.FlowInstaller
    port_base: int = 7000
    flows: list = field(default_factory=list)  # (link_key, sender host, receiver host, port)
    servers: list = field(default_factory=list)
    clients: list = field(default_factory=list)
    flow_names: list = field(default_factory=list)
    scenario: Scenario | None = None

    PIN_PRIORITY = 32768
    MIN_RATE_MBPS = 0.05
    DURATION_S = 86400

    def start(self) -> None:
        for index, (u, v) in enumerate(sorted(link_key(a, b) for a, b in self.spec.graph.edges)):
            a, b = self.spec.hosts_on(u)[0], self.spec.hosts_on(v)[0]
            self.flow_names += self.installer.install_path(self.spec.hosts[a].ip, self.spec.hosts[b].ip, [u, v],
                                                           prefix=f"bg_{u}_{v}", priority=self.PIN_PRIORITY)
            for direction, (src, dst) in enumerate(((a, b), (b, a))):
                self.flows.append(((u, v), src, dst, self.port_base + 2 * index + direction))
        log.info("Pinned %d background flows on %d links", len(self.flows), self.spec.graph.number_of_edges())

    @staticmethod
    def _terminate(procs) -> None:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _launch(self, key, src, dst, port, rate):
        return self.net.get(src).popen(
            ["iperf3", "-c", self.spec.hosts[dst].ip, "-p", str(port), "-u", "-b", f"{rate:.3f}M",
             "-t", str(self.DURATION_S)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def apply(self, scenario: Scenario, retries: int = 3) -> None:
        """Run the flows at the rates of `scenario` (both directions of a link get the same load).

        Servers are restarted with the clients: a server whose old client was killed on an overloaded
        link may never see the end of that test (its control messages sit in the full queue) and then
        refuses the next client as busy - silently leaving the most congested links idle. Every client
        is checked to be running a few seconds after start, and restarted if it is not."""
        self._terminate(self.clients)
        self._terminate(self.servers)
        self.servers = [self.net.get(dst).popen(["iperf3", "-s", "-p", str(port)], stdout=subprocess.DEVNULL,
                                                stderr=subprocess.DEVNULL) for _, _, dst, port in self.flows]
        time.sleep(1)
        wanted = [(flow, scenario.utilization[flow[0]] * self.spec.link_bw_mbps) for flow in self.flows]
        wanted = [(flow, rate) for flow, rate in wanted if rate >= self.MIN_RATE_MBPS]
        running = {flow: self._launch(*flow, rate) for flow, rate in wanted}
        for attempt in range(retries + 1):
            time.sleep(3)
            failed = [(flow, rate) for flow, rate in wanted if running[flow].poll() is not None]
            if not failed:
                break
            if attempt == retries:
                raise RuntimeError(f"{len(failed)} background flows would not start, e.g. {failed[0][0]}")
            log.warning("%d background flows exited at start; restarting them", len(failed))
            for flow, rate in failed:
                running[flow] = self._launch(*flow, rate)
        self.clients = list(running.values())
        self.scenario = scenario
        counts = Counter(class_of(u) for u in scenario.utilization.values())
        log.info("Scenario %s: links per class %s", scenario.seed,
                 ", ".join(f"{CLASS_NAMES[c]} {counts.get(c, 0)}" for c in range(len(CLASS_NAMES))))

    def stop(self) -> None:
        self._terminate(self.clients)
        self._terminate(self.servers)
        self.installer.remove(self.flow_names)
        self.servers, self.flows, self.flow_names = [], [], []


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topology", default="geant2", choices=sorted(TOPOLOGIES))
    parser.add_argument("--preview", action="store_true", help="class balance of 2,000 sampled scenarios")
    args = parser.parse_args()
    spec = build_topology(args.topology)
    s, r = spec.hosts[spec.sender], spec.hosts[spec.receiver]
    print(f"{spec.name}: {spec.graph.number_of_nodes()} switches, {spec.graph.number_of_edges()} links, "
          f"diameter {nx.diameter(spec.graph)}; Sender {spec.sender} on {s.switch}, Receiver {spec.receiver} on "
          f"{r.switch} ({nx.shortest_path_length(spec.graph, s.switch, r.switch)} hops)")
    if args.preview:
        from path_discovery import discover_paths
        candidates = [p.switches for p in discover_paths(spec.graph, s.switch, r.switch, "hop", 5).paths]
        counts = Counter(sample_scenario(spec.graph, seed, spec.path_hops).path_class(p)
                         for seed in range(2000) for p in candidates)
        total = sum(counts.values())
        print("Hop-count candidates per class: " + ", ".join(f"{CLASS_NAMES[c]} {counts[c] / total:.1%}"
                                                              for c in range(4)))


if __name__ == "__main__":
    main()
