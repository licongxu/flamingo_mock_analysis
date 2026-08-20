#!/usr/bin/env python3
"""IFFT harmonic W(ell) -> real-space W(theta) for MMF aperture integration.

    Y_ap(theta_c) = int d^2 theta  y(theta) W(theta - theta_c)

W(ell) = S(ell) / y(ell) with S = y_t^dagger N^{-1} d, built by
``compute_mmf_aperture_filter.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("SZIFI_ARRAY_BACKEND", "numpy")

import numpy as np

from flamingo_mock.szifi.paths import TILE_L_DEG, TILE_NX

HARMONIC = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues/"
    "mmf_aperture/field_0_theta5.npz"
)


def main() -> None:
    from szifi import maps

    if not HARMONIC.is_file():
        raise FileNotFoundError(f"missing {HARMONIC}")

    data = np.load(HARMONIC)
    w_fft = np.asarray(data["W_aperture"], dtype=np.complex128)
    dx = TILE_L_DEG / TILE_NX / 180.0 * np.pi
    pix = maps.pixel(TILE_NX, dx)
    w_theta = np.asarray(maps.get_ifft(w_fft, pix).real, dtype=np.float64)

    out = HARMONIC.with_name(
        f"field_{int(data['field_id'])}_theta{float(data['theta_500_arcmin']):.0f}_Wtheta.npz"
    )
    np.savez_compressed(
        out,
        W_theta=w_theta.astype(np.float32),
        W_fft=w_fft.astype(np.complex64),
        dx_rad=np.float64(dx),
        nx=np.int32(TILE_NX),
        ny=np.int32(TILE_NX),
        tile_l_deg=np.float64(TILE_L_DEG),
        centre_x=np.int32(TILE_NX // 2),
        centre_y=np.int32(TILE_NX // 2),
        theta_500_arcmin=np.float64(data["theta_500_arcmin"]),
        field_id=np.int32(data["field_id"]),
    )
    print(f"Wrote {out}  peak={np.max(np.abs(w_theta)):.6e}")


if __name__ == "__main__":
    main()
