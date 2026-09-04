"""Binary + C2-apodized mask from a homog iMMF catalogue (q>5).

Hole radius is max(4 theta_500, 2 FWHM) with FWHM=10 arcmin. Apodization
matches tsz_cnc_paper_plots: nmt.mask_apodization(..., 0.25, apotype="C2").

Default: L1_m9 fiducial (2509 detections, szifi_homog/L1_m9/catalogues/...).
--l2p8-test: legacy L2p8_m9 smoke-test catalogue (2364) at separate mask paths.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from hilc_prescriptions import (
    ALL_RUNS,
    catalogue_path,
    cluster_mask_apo,
    cluster_mask_binary,
)

NSIDE = 2048
Q_CUT = 5.0
FWHM_ARCMIN = 10.0
APOSIZE_DEG = 0.25
L2P8_TEST_CAT = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi_homog/catalogues"
    "/homog_immf_fullsky_splitA_immf_q5.npz"
)
ILC_DIR = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc")


def build_binary_mask(nside: int, lon_deg: np.ndarray, lat_deg: np.ndarray, radius_rad: np.ndarray) -> np.ndarray:
    mask = np.ones(hp.nside2npix(nside), dtype=np.float64)
    for lon, lat, rr in zip(lon_deg, lat_deg, radius_rad):
        vec = hp.ang2vec(float(lon), float(lat), lonlat=True)
        mask[hp.query_disc(nside, vec, float(rr))] = 0.0
    return mask


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prescription",
        choices=ALL_RUNS,
        default="L1_m9",
        help="szifi_homog/<name>/catalogues/fullsky_splitA_immf_q5.npz (default: L1_m9 fiducial)",
    )
    p.add_argument(
        "--l2p8-test",
        action="store_true",
        help="L2p8_m9 legacy smoke-test catalogue (2364 detections); separate mask paths",
    )
    args = p.parse_args()
    if args.l2p8_test:
        if args.prescription != "L1_m9":
            p.error("--l2p8-test cannot be combined with a non-default --prescription")
        cat = L2P8_TEST_CAT
        out_bin = ILC_DIR / "szifi_immf_q5_cluster_mask_l2p8test_nside2048.fits"
        out_apo = ILC_DIR / "szifi_immf_q5_cluster_mask_l2p8test_c2_025deg_nside2048.fits"
    else:
        cat = catalogue_path(args.prescription)
        out_bin = cluster_mask_binary(args.prescription)
        out_apo = cluster_mask_apo(args.prescription)

    det = np.load(cat)
    q = np.asarray(det["q_opt"], dtype=np.float64)
    sel = q > Q_CUT
    lon = np.asarray(det["lon"], dtype=np.float64)[sel]
    lat = np.asarray(det["lat"], dtype=np.float64)[sel]
    th500 = np.asarray(det["theta_500"], dtype=np.float64)[sel]
    radius = np.maximum(
        4.0 * np.deg2rad(th500 / 60.0),
        2.0 * np.deg2rad(FWHM_ARCMIN / 60.0),
    )
    print(f"catalogue={cat}")
    print(f"n_q>{Q_CUT:g} = {int(sel.sum())} / {q.size}")
    print(
        f"radius arcmin: min={np.rad2deg(radius.min())*60:.2f}  "
        f"median={np.rad2deg(np.median(radius))*60:.2f}  "
        f"max={np.rad2deg(radius.max())*60:.2f}"
    )

    mask_bin = build_binary_mask(NSIDE, lon, lat, radius)
    print("C2 apodization ...", flush=True)
    mask_apo = nmt.mask_apodization(mask_bin, APOSIZE_DEG, apotype="C2")
    fsky_raw = float(mask_bin.mean())
    fsky_eff = float(np.mean(mask_apo**2))
    soft = float(np.mean((mask_apo > 0.0) & (mask_apo < 1.0)))
    print(
        f"f_sky raw={fsky_raw:.4f}  eff=<W^2>={fsky_eff:.4f}  "
        f"soft-edge={soft:.4f}"
    )

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    hp.write_map(str(out_bin), mask_bin, nest=False, overwrite=True, dtype=np.float64)
    hp.write_map(str(out_apo), mask_apo, nest=False, overwrite=True, dtype=np.float64)
    print("wrote", out_bin)
    print("wrote", out_apo)


if __name__ == "__main__":
    main()
