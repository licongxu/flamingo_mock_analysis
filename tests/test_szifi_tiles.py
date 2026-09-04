"""Unit tests for SZiFi tile selection."""

import healpy as hp

from flamingo_mock.szifi.tiles import select_all_tile_ids, select_pilot_tile_ids


def test_pilot_tiles_high_latitude():
    ids = select_pilot_tile_ids(n=4, b_min_deg=40.0)
    assert len(ids) == 4
    assert len(set(ids)) == 4
    for i in ids:
        _, b = hp.pix2ang(8, i, lonlat=True)
        assert abs(b) >= 40.0


def test_all_tile_ids_full_sky():
    ids = select_all_tile_ids()
    assert ids == list(range(hp.nside2npix(8)))
    assert len(ids) == 768
