"""
topology_builder.py - the AIRSDN evaluation network: NSFNET in Mininet.

Reproduces the scenario of the paper (Section 4.2, Fig. 14) and of the paper's original experiment scripts:
  * 14 switches (s0-s13) and 21 links of the NSFNET topology, 25 ms / 10 Mbps each
  * 20 hosts: h0 (video client, on s0), h8 (video server, on s9) and 18 traffic generators
  * 9 background iperf3 flows pinned onto single links: two 100M flows on each of the links
    s3-s8, s1-s7, s11-s10 and s2-s5, and one 3M flow on s3-s4

The link order matches the original script, so Mininet assigns the same port numbers and
hop-count path ranking (Yen) gives the same candidate order as Floodlight.

Offline check (no root needed):
    python airsdn_reproduction/topology_builder.py
"""
from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field

import networkx as nx
from mininet.topo import Topo

log = logging.getLogger(__name__)

LINK_BW_MBPS = 10
LINK_DELAY = "25ms"
QUEUE_SIZE = 1000

SWITCH_LINKS = [(0, 1), (0, 2), (0, 3), (1, 7), (1, 2), (2, 5), (3, 4), (3, 8), (4, 5), (4, 6), (5, 12),
                (5, 13), (6, 7), (7, 10), (8, 9), (8, 11), (9, 10), (9, 12), (10, 11), (10, 13), (11, 12)]
# host index -> switch index (h<i> has IP 10.0.0.<i+1>), in the original link order
HOST_SWITCH = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 3, 1, 8, 7, 11, 2, 10, 5, 3]
CLIENT, SERVER = "h0", "h8"          # video receiver / sender (10.0.0.1 / 10.0.0.9)

# (sender, receiver, iperf3 rate, pinned switch path) - the paper's original experiment scripts generateTraffic/setTrafficPaths
BACKGROUND_FLOWS = [
    ("h7", "h3", "100M", ["s8", "s3"]), ("h13", "h11", "100M", ["s8", "s3"]),
    ("h6", "h1", "100M", ["s7", "s1"]), ("h14", "h12", "100M", ["s7", "s1"]),
    ("h9", "h10", "100M", ["s10", "s11"]), ("h17", "h15", "100M", ["s10", "s11"]),
    ("h5", "h2", "100M", ["s5", "s2"]), ("h18", "h16", "100M", ["s5", "s2"]),
    ("h4", "h19", "3M", ["s4", "s3"]),
]


@dataclass
class HostInfo:
    name: str
    ip: str
    switch: str
    switch_port: int


@dataclass
class TopologySpec:
    topo: Topo
    graph: nx.Graph                       # switch graph; edges carry the port on each side
    hosts: dict[str, HostInfo]
    switch_dpids: dict[str, str]          # switch name -> Floodlight DPID (colon form)
    link_bw_mbps: float = LINK_BW_MBPS

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
        """Root-namespace interface of the switch port facing `host` (e.g. s0-eth4)."""
        info = self.hosts[host]
        return f"{info.switch}-eth{info.switch_port}"


def dpid(index: int) -> str:
    # The original script numbers DPIDs 01..14 in decimal digits (s9 -> ...:10)
    return f"{index + 1:02d}".rjust(16, "0")


class NsfnetTopo(Topo):
    def build(self):
        for i in range(14):
            self.addSwitch(f"s{i}", dpid=dpid(i), protocols="OpenFlow13")
        options = dict(bw=LINK_BW_MBPS, delay=LINK_DELAY, max_queue_size=QUEUE_SIZE, use_htb=True)
        for a, b in SWITCH_LINKS:
            self.addLink(f"s{a}", f"s{b}", **options)
        for i, sw in enumerate(HOST_SWITCH):
            host = self.addHost(f"h{i}", ip=f"10.0.0.{i + 1}/8")
            self.addLink(f"s{sw}", host, **options)


def build_nsfnet() -> TopologySpec:
    topo = NsfnetTopo()
    graph = nx.Graph()
    graph.add_nodes_from(f"s{i}" for i in range(14))
    for a, b in SWITCH_LINKS:
        u, v = f"s{a}", f"s{b}"
        port_u, port_v = topo.port(u, v)
        graph.add_edge(u, v, ports={u: port_u, v: port_v})
    hosts = {}
    for i, sw in enumerate(HOST_SWITCH):
        switch_port, _ = topo.port(f"s{sw}", f"h{i}")
        hosts[f"h{i}"] = HostInfo(f"h{i}", f"10.0.0.{i + 1}", f"s{sw}", switch_port)
    dpids = {f"s{i}": ":".join(dpid(i)[j:j + 2] for j in range(0, 16, 2)) for i in range(14)}
    return TopologySpec(topo=topo, graph=graph, hosts=hosts, switch_dpids=dpids)


def start_network(spec: TopologySpec, controller_ip="127.0.0.1", controller_port=6653):
    """Start NSFNET on the remote Floodlight controller (needs root)."""
    from mininet.link import TCLink
    from mininet.net import Mininet
    from mininet.node import OVSKernelSwitch, RemoteController

    net = Mininet(topo=spec.topo, switch=OVSKernelSwitch, link=TCLink, controller=None,
                  autoSetMacs=True, build=False, waitConnected=True)
    net.addController("c1", controller=RemoteController, ip=controller_ip, port=controller_port,
                      protocols="OpenFlow13")
    try:
        net.build()
        net.start()
    except Exception:
        net.stop()
        raise
    return net


def stop_network(net) -> None:
    """Stop Mininet, interrupting commands still running in host shells first."""
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


@dataclass
class BackgroundTraffic:
    """The paper's 9 background iperf3 flows, each pinned onto its link with static entries."""
    net: object
    spec: TopologySpec
    installer: object                     # telemetry_and_model.FlowInstaller
    duration_s: int = 3600
    procs: list = field(default_factory=list)
    flow_names: list = field(default_factory=list)

    PRIORITY = 32768
    PORT = 5555

    def start(self) -> None:
        for sender, receiver, rate, path in BACKGROUND_FLOWS:
            s_ip, r_ip = self.spec.hosts[sender].ip, self.spec.hosts[receiver].ip
            self.flow_names += self.installer.install_path(s_ip, r_ip, path, prefix=f"bg_{sender}_{receiver}",
                                                           priority=self.PRIORITY)
            # as in the paper's original experiment scripts: the sender runs the iperf3 server, the receiver the client in
            # reverse mode, so data flows sender -> receiver
            self.procs.append(self.net.get(sender).popen(["iperf3", "-s", "-p", str(self.PORT)],
                                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            time.sleep(0.3)
            self.procs.append(self.net.get(receiver).popen(
                ["iperf3", "-c", s_ip, "-p", str(self.PORT), "-b", rate, "-R", "-t", str(self.duration_s)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        log.info("Started %d background flows", len(BACKGROUND_FLOWS))

    def stop(self) -> None:
        for proc in self.procs:
            if proc.poll() is None:
                proc.terminate()
        self.installer.remove(self.flow_names)
        self.procs, self.flow_names = [], []


if __name__ == "__main__":
    spec = build_nsfnet()
    print(f"NSFNET: {spec.graph.number_of_nodes()} switches, {spec.graph.number_of_edges()} links, "
          f"{len(spec.hosts)} hosts; client {CLIENT}@{spec.hosts[CLIENT].switch}, "
          f"server {SERVER}@{spec.hosts[SERVER].switch}")
