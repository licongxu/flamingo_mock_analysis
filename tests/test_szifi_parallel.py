"""Tests for half-machine parallelism caps."""

from flamingo_mock.szifi.run import half_machine_pool_limits


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


def test_half_machine_pool_limits_many_workers(monkeypatch):
    monkeypatch.setattr("flamingo_mock.szifi.run.os.cpu_count", lambda: 192)
    workers, threads = half_machine_pool_limits(24, threads_per_worker=4)
    assert workers == 24
    assert threads == 4
    assert workers * threads <= 96
