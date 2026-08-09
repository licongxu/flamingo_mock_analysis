"""Tests for flamingo_mock.ilc.config: YAML generation and pyILC parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
from pyilc.input import ILCInfo

from flamingo_mock.ilc.config import ILCConfig, pyilc_param_dict_file
from flamingo_mock.ilc.paths import ILCPaths

REPO = Path(__file__).resolve().parents[1]
HILC_SPLIT_A = REPO / "configs" / "hilc_y_flamingo_npipe_splitA.yml"
HILC_SPLIT_B = REPO / "configs" / "hilc_y_flamingo_npipe_splitB.yml"
NILC_SPLIT_A = REPO / "configs" / "nilc_y_flamingo_npipe_splitA.yml"


def test_param_dict_file_from_installed_pyilc():
    assert pyilc_param_dict_file().is_file()


def test_config_writer_roundtrip_hilc(tmp_path):
    cfg = ILCConfig(method="hilc", split="A", paths=ILCPaths())
    out = cfg.write(tmp_path / "hilc.yml")
    info = ILCInfo(str(out))
    assert info.ELLMAX >= 3000
    assert info.N_side >= 2048
    assert info.ILC_preserved_comp == "tSZ"
    assert info.N_deproj == 0
    assert info.wavelet_type == "TopHatHarmonic"
    assert info.perform_ILC_at_beam == pytest.approx(5.0)
    assert list(info.beam_FWHM_arcmin) == pytest.approx([9.66, 7.22, 4.92])
    assert info.ilc_backend == "jax"
    assert info.N_freqs == 3
    assert len(info.freq_map_files) == 3
    for f in info.freq_map_files:
        assert "npipe_splitA" in f
        assert "100GHz" in f or "143GHz" in f or "353GHz" in f


def test_config_writer_roundtrip_nilc(tmp_path):
    cfg = ILCConfig(method="nilc", split="A", paths=ILCPaths())
    out = cfg.write(tmp_path / "nilc.yml")
    info = ILCInfo(str(out))
    assert info.wavelet_type == "GaussianNeedlets"
    assert info.N_scales == 10
    assert info.ELLMAX >= 3000
    assert info.ILC_preserved_comp == "tSZ"
    assert info.perform_ILC_at_beam == pytest.approx(5.0)


def test_splits_differ_only_in_maps_and_output(tmp_path):
    a = ILCConfig(method="hilc", split="A", paths=ILCPaths()).to_dict()
    b = ILCConfig(method="hilc", split="B", paths=ILCPaths()).to_dict()
    assert a["freq_map_files"] != b["freq_map_files"]
    assert all("splitA" in f for f in a["freq_map_files"])
    assert all("splitB" in f for f in b["freq_map_files"])
    assert a["output_dir"] != b["output_dir"]
    for key in ("ELLMAX", "N_side", "beam_FWHM_arcmin", "ilc_backend"):
        assert a[key] == b[key]


def test_tracked_repo_configs_parse_and_match_writer():
    """The tracked configs/ YAMLs must stay in sync with the package writer."""
    for path, method, split in (
        (HILC_SPLIT_A, "hilc", "A"),
        (HILC_SPLIT_B, "hilc", "B"),
        (NILC_SPLIT_A, "nilc", "A"),
    ):
        assert path.is_file(), f"{path} missing — run: flamingo-ilc config"
        on_disk = ILCInfo(str(path))
        fresh = ILCConfig(method=method, split=split, paths=ILCPaths())
        assert on_disk.ELLMAX == fresh.ellmax
        assert on_disk.ilc_backend == fresh.ilc_backend
        assert list(on_disk.freq_map_files) == [
            str(p) for p in map(Path, fresh.to_dict()["freq_map_files"])
        ]


def test_noise_split_input_maps_exist_when_prepared():
    info = ILCInfo(str(HILC_SPLIT_A))
    present = [Path(f) for f in info.freq_map_files if Path(f).is_file()]
    if len(present) < 3:
        pytest.skip("noise-split inputs not prepared yet (flamingo-ilc prepare)")
    for f in present:
        assert f.stat().st_size > 1_000_000
    info_b = ILCInfo(str(HILC_SPLIT_B))
    for f in info_b.freq_map_files:
        p = Path(f)
        if not p.is_file():
            pytest.skip("split B maps missing")
        assert p.stat().st_size > 1_000_000
