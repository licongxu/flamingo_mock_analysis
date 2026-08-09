"""tSZ component: Compton-y map -> thermodynamic temperature maps dT(nu)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import TSZ_FILE, MockConfig
from .io import copy_or_link, load_flamingo_map, write_map
from .spectral import y_to_delta_T_uK


def load_compton_y(cfg: MockConfig) -> np.ndarray:
    """Load the lensed Compton-y map (dimensionless)."""
    return load_flamingo_map(cfg.data_dir / TSZ_FILE, cfg.nside)


def archive_compton_y(
    cfg: MockConfig, out_dir: Path | None = None, *, use_symlink: bool = True
) -> Path:
    """Copy or link the lensed Compton-y map from the FLAMINGO release."""
    out_dir = out_dir or cfg.raw_dir / "tsz" / "tsz"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"compton_y_nside{cfg.nside}.fits"
    if not dst.exists():
        print("tSZ: archiving lensed Compton-y map...")
        copy_or_link(cfg.data_dir / TSZ_FILE, dst, use_symlink=use_symlink)
    else:
        print(f"tSZ: reusing {dst.name}")
    return dst


def make_tsz_maps(
    cfg: MockConfig, y: np.ndarray | None = None, out_dir: Path | None = None
) -> dict[float, np.ndarray]:
    """Build dT_tSZ(nu) [uK_CMB] for every frequency in the config.

    Also writes the Compton-y map itself (needed as ILC ground truth).
    """
    out_dir = out_dir or cfg.raw_dir / "tsz"
    if y is None:
        print("tSZ: loading lensed Compton-y map...")
        y = load_compton_y(cfg)
    print(f"tSZ: y mean={y.mean():.4e}, std={y.std():.4e}")

    write_map(
        out_dir / f"compton_y_nside{cfg.nside}.fits",
        y,
        unit="Compton_y",
        extra=[("COMP", "tSZ_y")],
    )

    maps = {}
    for nu in cfg.frequencies:
        dt = y_to_delta_T_uK(y, nu, t_cmb=cfg.t_cmb)
        maps[nu] = dt
        write_map(
            out_dir / f"tSZ_deltaT_{nu:.0f}GHz_nside{cfg.nside}.fits",
            dt,
            unit="uK_CMB",
            freq=nu,
            extra=[("COMP", "tSZ"), ("KERNEL", "nonrel f(x)=x*coth(x/2)-4")],
        )
        print(f"  tSZ {nu:g} GHz: std={dt.std():.3e} uK")
    return maps
