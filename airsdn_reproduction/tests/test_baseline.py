"""Offline tests for the baseline modules (run: .venv/bin/python -m pytest airsdn_reproduction/tests)."""
import itertools
import os
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from path_discovery import discover_paths, yen_hop_count  # noqa: E402
from topology_builder import BACKGROUND_FLOWS, CLIENT, SERVER, build_nsfnet  # noqa: E402

SPEC = build_nsfnet()


def test_nsfnet_shape_and_ports():
    assert SPEC.graph.number_of_nodes() == 14 and SPEC.graph.number_of_edges() == 21
    assert len(SPEC.hosts) == 20
    assert (SPEC.hosts[CLIENT].switch, SPEC.hosts[SERVER].switch) == ("s0", "s9")
    # same port numbers as the paper's original experiment scripts: the client is on s0-eth4 (NFStream listens there)
    assert SPEC.access_interface(CLIENT) == "s0-eth4"
    assert SPEC.port("s3", "s8") == 3 and SPEC.port("s8", "s3") == 1
    assert SPEC.switch_dpids["s9"] == "00:00:00:00:00:00:00:10"


def test_candidates_match_floodlight_and_the_paper():
    paths = discover_paths(SPEC.graph, "s0", "s9").paths
    assert ["-".join(p) for p in paths] == ["s0-s3-s8-s9", "s0-s1-s7-s10-s9", "s0-s2-s5-s12-s9",
                                            "s0-s3-s4-s5-s12-s9", "s0-s3-s8-s11-s10-s9"]


@pytest.mark.parametrize("src,dst", [("s0", "s9"), ("s6", "s11"), ("s13", "s3")])
def test_yen_hop_counts_match_networkx(src, dst):
    ours = [len(p) - 1 for p in yen_hop_count(SPEC.graph, src, dst, 8)]
    reference = [len(p) - 1 for p in itertools.islice(nx.shortest_simple_paths(SPEC.graph, src, dst), 8)]
    assert ours == reference


def test_background_flows_are_on_existing_links():
    for sender, receiver, _, path in BACKGROUND_FLOWS:
        assert SPEC.hosts[sender].switch == path[0] and SPEC.hosts[receiver].switch == path[-1]
        assert SPEC.graph.has_edge(*path)
