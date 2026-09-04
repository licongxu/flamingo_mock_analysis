"""Unit tests for footprint tile selection."""

import numpy as np
import healpy as hp

from flamingo_mock.szifi.paths import SZiFiPaths
from flamingo_mock.szifi.tiles import select_footprint_tile_ids


def test_footprint_tile_count_reasonable():
    paths = SZiFiPaths()
    ids = select_footprint_tile_ids(paths.masks_fits, min_ftile=0.3)
    assert 400 <= len(ids) <= 600
    assert ids == sorted(ids)
    assert ids[0] >= 0
    assert ids[-1] < hp.nside2npix(8)
