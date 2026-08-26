#!/usr/bin/env python3
"""CIB products, one folder per L1_m9 Yang26 prescription.

  components/cib/L1_m9/          fiducial hydro (D3A)
  components/cib/fgas-8sigma/
  components/cib/Mstar-1sigma/
  components/cib/LS8/
  components/cib/test/           L2p8 demo (parked)

Each folder has CIB_I_{217,353,545,857} (Jy/sr, symlink to the HDF5)
and CIB_deltaT_{100,143,217,353,545,857} (uK_CMB).
"""
from __future__ import annotations

from pathlib import Path

from flamingo_mock import cib
from flamingo_mock.config import PLANCK_FREQUENCIES_GHZ, MockConfig

COMP = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/components")
CIB = COMP / "cib"
PRESCRIPTIONS = ("L1_m9", "fgas-8sigma", "Mstar-1sigma", "LS8")

CIB_NAMES = [
    *[f"CIB_deltaT_{nu}GHz_nside4096.fits" for nu in (100, 143, 217, 353, 545, 857)],
    *[f"CIB_I_{nu}GHz_nside4096.hdf5" for nu in (217, 353, 545, 857)],
]
CIB_HDF5 = {
    217: "lensed_CIB_rot_BANDPASS_F217_three_params_same_rot.hdf5",
    353: "lensed_CIB_rot_BANDPASS_F353_three_params_same_rot.hdf5",
    545: "lensed_CIB_rot_BANDPASS_F545_three_params_same_rot.hdf5",
    857: "lensed_CIB_rot_BANDPASS_F857_three_params_same_rot.hdf5",
}


def relocate_fiducial() -> None:
    dest = CIB / "L1_m9"
    dest.mkdir(parents=True, exist_ok=True)
    for name in CIB_NAMES:
        src = CIB / name
        if not src.exists() and not src.is_symlink():
            continue
        target = dest / name
        if target.exists() or target.is_symlink():
            if src.resolve() == target.resolve():
                continue
            print(f"skip {name}: already in L1_m9/")
            continue
        src.rename(target)
        print(f"moved cib/{name} -> cib/L1_m9/")


def complete(run: str) -> bool:
    d = CIB / run
    return all((d / n).exists() or (d / n).is_symlink() for n in CIB_NAMES)


def build_cib(run: str) -> None:
    cfg = MockConfig(
        data_dir=COMP / "L1_m9" / run,
        frequencies=PLANCK_FREQUENCIES_GHZ,
        nside=4096,
        cib_files=dict(CIB_HDF5),
    )
    out = CIB / run
    print(f"\n=== CIB {run} ===")
    print("data:", cfg.data_dir)
    cib.copy_released_cib_intensity(cfg, out_dir=out, use_symlink=True)
    cib.make_cib_maps(cfg, out_dir=out)


def main() -> None:
    relocate_fiducial()
    for run in PRESCRIPTIONS:
        if complete(run):
            print(f"{run}: CIB products already present, skip")
            continue
        build_cib(run)
    print("\nCIB prescriptions:")
    for run in PRESCRIPTIONS:
        tag = "  (fiducial hydro)" if run == "L1_m9" else ""
        print(f"  {CIB / run}{tag}")
    print("Done.")


if __name__ == "__main__":
    main()
