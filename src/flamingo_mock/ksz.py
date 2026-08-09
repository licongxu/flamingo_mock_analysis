"""kSZ component: Doppler-b map -> thermodynamic temperature fluctuation.

The kSZ signal is frequency independent in thermodynamic units:
dT_kSZ / T_CMB = -b, so a single map serves all frequencies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import KSZ_FILE, MockConfig
from .io import load_flamingo_map, write_map
from .spectral import b_to_delta_T_uK


def load_doppler_b(cfg: MockConfig) -> np.ndarray:
    """Load the lensed Doppler-b map (dimensionless, dT/T_CMB = -b)."""
    return load_flamingo_map(cfg.data_dir / KSZ_FILE, cfg.nside)


def make_ksz_map(
    cfg: MockConfig, b: np.ndarray | None = None, out_dir: Path | None = None
) -> np.ndarray:
    """Build dT_kSZ [uK_CMB] (frequency independent in thermodynamic units)."""
    out_dir = out_dir or cfg.raw_dir
    if b is None:
        print("kSZ: loading lensed Doppler-b map...")
        b = load_doppler_b(cfg)
    print(f"kSZ: b mean={b.mean():.4e}, std={b.std():.4e}")

    write_map(
        out_dir / f"doppler_b_nside{cfg.nside}.fits",
        b,
        unit="Doppler_b",
        extra=[("COMP", "kSZ_b")],
    )

    dt = b_to_delta_T_uK(b, t_cmb=cfg.t_cmb)
    write_map(
        out_dir / f"kSZ_deltaT_nside{cfg.nside}.fits",
        dt,
        unit="uK_CMB",
        extra=[("COMP", "kSZ"), ("KERNEL", "dT = -T_CMB * b (freq indep.)")],
    )
    print(f"  kSZ dT: std={dt.std():.3e} uK (same at all frequencies)")
    return dt
