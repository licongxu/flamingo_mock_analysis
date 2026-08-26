#!/usr/bin/env python3
"""Lensed primary CMB at FLAMINGO LS8 cosmology (Yang et al. 2026 Table 1).

CAMB unlensed C_ell^TT with LS8 parameters, deflected by pixell using the
Yang26 LS8 kappa map. Writes alongside the existing D3A CMB products
(does not overwrite them).
"""
from __future__ import annotations

from pathlib import Path

from flamingo_mock.cmb import make_lensed_cmb
from flamingo_mock.config import COSMOLOGY_LS8, MockConfig

KAPPA = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/"
    "L1_m9/LS8/CMB_lensing_rot_same_rot.hdf5"
)
OUT = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/cmb")


def main() -> None:
    cfg = MockConfig(nside=4096, seed=42, cosmology=dict(COSMOLOGY_LS8))
    print("cosmo LS8:", cfg.cosmology)
    print("kappa:", KAPPA)
    print("out:", OUT)
    t = make_lensed_cmb(
        cfg, out_dir=OUT, kappa_path=KAPPA, name_suffix="LS8"
    )
    print(f"CMB LS8 lensed: std={t.std():.2f} uK  nside={cfg.nside}")


if __name__ == "__main__":
    main()
