"""Tests for half-machine parallelism caps and resume identity."""

import json

import numpy as np

from flamingo_mock.szifi.paths import SZiFiPaths
from flamingo_mock.szifi.run import _partial_matches_batch, half_machine_pool_limits
from flamingo_mock.szifi.true_snr import _tile_checkpoint_path


def test_half_machine_pool_limits_respects_half_cpus(monkeypatch):
    monkeypatch.setattr("flamingo_mock.szifi.run.os.cpu_count", lambda: 192)
    workers, threads = half_machine_pool_limits(None)
    assert workers * threads <= 96
    assert workers <= 8
    assert workers >= 1


def test_half_machine_pool_limits_explicit_workers(monkeypatch):
    monkeypatch.setattr("flamingo_mock.szifi.run.os.cpu_count", lambda: 192)
    workers, threads = half_machine_pool_limits(6)
    assert workers == 6
    assert workers * threads <= 96


def test_partial_matches_batch_requires_matching_field_ids(tmp_path):
    part = tmp_path / "batch_0000_q5.npz"
    np.savez_compressed(part, q_opt=np.array([5.0]))
    assert not _partial_matches_batch(part, [1, 2])
    part.with_suffix(".json").write_text(json.dumps({"field_ids": [1, 2]}))
    assert _partial_matches_batch(part, [1, 2])
    assert not _partial_matches_batch(part, [9, 2])


def test_true_snr_checkpoint_keyed_by_split_and_parent(tmp_path):
    paths = SZiFiPaths(out_root=tmp_path)
    a = _tile_checkpoint_path(
        paths, 3, split="A", truth_csv="truth.csv", z_max=1.0, q_ap_min=2.0
    )
    b = _tile_checkpoint_path(
        paths, 3, split="B", truth_csv="truth.csv", z_max=1.0, q_ap_min=2.0
    )
    assert a != b
    assert "splitA" in str(a) and "splitB" in str(b)
