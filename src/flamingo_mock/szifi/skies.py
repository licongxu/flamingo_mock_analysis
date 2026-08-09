"""Load multi-frequency total skies for SZiFi (coadd + beam + NPIPE)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .paths import FREQS_GHZ, SZiFiPaths

UK_PER_K = 1.0e6


def load_total_map_uK(
    paths: SZiFiPaths,
    split: str,
    freq_ghz: int,
) -> np.ndarray:
    """Load one total sky map and convert K_CMB → µK_CMB."""
    import healpy as hp

    path = paths.total_map_path(split, freq_ghz)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing total map {path}. Rebuild coadd+beam+noise or point "
            f"SZiFiPaths.total_maps_dir at existing products."
        )
    m = np.asarray(hp.read_map(str(path), field=0, dtype=np.float64))
    return m * UK_PER_K


def load_total_maps_uK(
    paths: SZiFiPaths,
    split: str = "A",
    freqs: tuple[int, ...] = FREQS_GHZ,
) -> dict[int, np.ndarray]:
    """Load all frequency channels [µK_CMB] for one NPIPE split."""
    return {int(f): load_total_map_uK(paths, split, int(f)) for f in freqs}


def stack_maps_nxnxnf(
    maps_uK: dict[int, np.ndarray],
    freqs: tuple[int, ...] = FREQS_GHZ,
) -> list[np.ndarray]:
    """Return channel maps in SZiFi frequency order."""
    return [maps_uK[int(f)] for f in freqs]
