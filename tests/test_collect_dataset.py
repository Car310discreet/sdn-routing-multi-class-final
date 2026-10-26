"""Offline tests of the dataset collector: probe choice, labelling, quotas and trimming (no Mininet/Floodlight)."""
import os
import random
import shutil
import sys
import types
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "routing"))
import collect_dataset  # noqa: E402
from telemetry_and_model import FEATURE_NAMES  # noqa: E402
from topology_builder import FATTREE_EDGE_SWITCHES, build_topology, sample_scenario  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


class FakeInstaller:
    def install_path(self, src_ip, dst_ip, path, prefix, priority=0, ip_protos=None):
        assert ip_protos == [1, 6]
        return [prefix]

    def remove(self, names):
        pass


class FakeTelemetry:
    def __init__(self, scenario):
        self.scenario = scenario

    def measure(self, src, dst, path, snapshot):
        load = self.scenario.path_utilization(path)
        features = {name: 0.0 for name in FEATURE_NAMES}
        features.update(hop_count=len(path) - 1.0, path_max_link_util_pct=100 * min(load, 1.0),
                        packet_loss_pct=100.0 if load > 1.1 else 0.0)
        return features


def test_epoch_probes_sender_candidates_and_random_pairs():
    for name in ("geant2", "fattree"):
        spec = build_topology(name)
        candidates = collect_dataset.hop_candidates(spec.graph, spec.hosts[spec.sender].switch,
                                                    spec.hosts[spec.receiver].switch)
        probes = collect_dataset.epoch_paths(spec, random.Random(1), candidates, 3)
        assert len(probes) == 8 and all((s, d) == (spec.sender, spec.receiver) for s, d, _ in probes[:5])
        for src, dst, path in probes[5:]:
            assert spec.hosts[src].switch == path[0] and spec.hosts[dst].switch == path[-1]
            if name == "fattree":
                assert path[0] in FATTREE_EDGE_SWITCHES and path[-1] in FATTREE_EDGE_SWITCHES


def test_measure_epoch_labels_with_ground_truth_and_respects_quotas():
    spec = build_topology("geant2")
    candidates = collect_dataset.hop_candidates(spec.graph, spec.hosts[spec.sender].switch,
                                                spec.hosts[spec.receiver].switch)
    args = types.SimpleNamespace(extra_pairs=3)
    total = Counter()
    for seed in range(20):
        scenario = sample_scenario(spec.graph, seed, spec.path_hops, target_class=seed % 4)
        rows = collect_dataset.measure_epoch(spec, random.Random(seed), candidates, args, scenario, total, 6,
                                             FakeInstaller(), FakeTelemetry(scenario), snapshot=None)
        for _, _, path, _, label in rows:
            assert label == scenario.path_class(path)
            total[label] += 1
    assert all(count <= 6 for count in total.values()) and len(total) == 4


def test_trim_reproduces_the_report_fattree_dataset(tmp_path):
    full = tmp_path / "fattree.csv"
    shutil.copyfile(os.path.join(DATA, "fattree_dataset_full.csv"), full)
    collect_dataset.trim(str(full))
    trimmed, reference = pd.read_csv(full), pd.read_csv(os.path.join(DATA, "fattree_dataset.csv"))
    assert len(pd.read_csv(tmp_path / "fattree_full.csv")) == 912
    assert trimmed["traffic_class"].value_counts().tolist() == [224] * 4
    pd.testing.assert_frame_equal(trimmed.reset_index(drop=True), reference)
