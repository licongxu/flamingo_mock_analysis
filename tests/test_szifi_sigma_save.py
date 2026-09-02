"""Per-tile sigma_y0 persistence during SZiFi runs."""

from types import SimpleNamespace

import numpy as np

from flamingo_mock.szifi.paths import SZiFiPaths
from flamingo_mock.szifi.run import save_per_tile_sigma, sigma_per_tile_dir


def test_sigma_per_tile_dir_layout():
    paths = SZiFiPaths(out_root="/tmp/szifi_test_root")
    d = sigma_per_tile_dir(paths, method="immf", split="A")
    assert d.name == "sigma_per_tile_immf_splitA"
    assert d.parent.name == "catalogues"


def test_save_per_tile_sigma_writes_find1_and_noit(tmp_path):
    theta = np.array([1.0, 2.0, 5.0])
    results = {
        3: SimpleNamespace(
            sigma_vec={
                "find_0": np.array([0.1, 0.2, 0.3]),
                "find_1": np.array([0.11, 0.21, 0.31]),
            }
        ),
        7: SimpleNamespace(sigma_vec={"find_0": np.array([0.4, 0.5, 0.6])}),
    }
    n = save_per_tile_sigma(results, theta, tmp_path)
    assert n == 2
    assert np.allclose(np.load(tmp_path / "theta_500_arcmin.npy"), theta)
    assert np.allclose(np.load(tmp_path / "field_3.npy"), [0.11, 0.21, 0.31])
    assert np.allclose(np.load(tmp_path / "field_3_noit.npy"), [0.1, 0.2, 0.3])
    assert np.allclose(np.load(tmp_path / "field_7.npy"), [0.4, 0.5, 0.6])
    assert np.allclose(np.load(tmp_path / "field_7_noit.npy"), [0.4, 0.5, 0.6])
