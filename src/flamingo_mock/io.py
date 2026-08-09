"""HEALPix map I/O helpers for the FLAMINGO integrated maps and products."""

from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np

from .config import NSIDE_NATIVE


def load_flamingo_map(path: Path, nside_out: int | None = None) -> np.ndarray:
    """Load a FLAMINGO integrated map (Nside=4096 RING FITS, 'data' column).

    Parameters
    ----------
    path
        FITS file of the integrated map.
    nside_out
        If given and smaller than the native Nside, downgrade with
        ``hp.ud_grade`` (for quick tests / previews).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    m = hp.read_map(str(path), dtype=np.float64)
    if m.size != 12 * NSIDE_NATIVE**2:
        raise ValueError(
            f"{path.name}: expected {12 * NSIDE_NATIVE**2} pixels "
            f"(Nside={NSIDE_NATIVE}), got {m.size}"
        )
    if nside_out is not None and nside_out != NSIDE_NATIVE:
        m = hp.ud_grade(m, nside_out)
    return m


def write_map(
    path: Path,
    m: np.ndarray,
    unit: str,
    freq: float | None = None,
    extra: list[tuple] | None = None,
    dtype=np.float32,
) -> Path:
    """Write a HEALPix FITS map with a minimal self-documenting header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = [("UNIT", unit), ("NSIDE", int(hp.get_nside(m)))]
    if freq is not None:
        hdr.append(("FREQ", float(freq), "GHz"))
    if extra:
        hdr.extend(extra)
    hp.write_map(
        str(path),
        np.asarray(m, dtype=dtype),
        overwrite=True,
        dtype=dtype,
        column_names=["TEMPERATURE"] if "K" in unit else ["DATA"],
        extra_header=hdr,
    )
    print(f"  wrote {path} ({path.stat().st_size / 1e9:.2f} GB)")
    return path
