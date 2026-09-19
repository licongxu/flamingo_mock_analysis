"""Masked tSZ PS of the L1_m9 truth y-map using the sciMMF catalogue.

Same estimator as flamingo_repo/scripts/compute_l1_m9_masked_ps_alpha_fixed_1p12.py:
disc r = max(4 theta_500, 2*10'), C2 0.25 deg, mask-weighted monopole,
NaMaster nlb=1 lmax=10000, pixwin deconvolved, no beam.

Run::

    python scripts/compute_l1_m9_scimmf_masked_ps.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt

MAP_FILE = Path("/rds/rds-lxu/flamingo/L1_m9/maps/y_unlensed_L1_m9_lc0_nside4096.fits")
CAT_FILE = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi_homog/L1_m9"
    "/catalogues/szifi_jax_scimmf_splitA_immf_q5.npz"
)
OUT_DIR = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi_homog/L1_m9"
    "/ps_masked_scimmf"
)

Q_CUTS = [50.0, 20.0, 10.0, 6.0]
CUT_TAGS = ["qgt50", "qgt20", "qgt10", "qgt6"]
FWHM_ARCMIN = 10.0
R_MULT = 4.0
APOSIZE_DEG = 0.25
APOTYPE = "C2"
LMAX = 10000
ELL_MIN = np.array(
    [9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835]
)
ELL_MAX = np.array(
    [12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835, 1085]
)
ELL_EFF = np.array(
    [10.0, 13.5, 18.0, 23.5, 30.5, 40.0, 52.5, 68.5, 89.5, 117.0, 152.5, 198.0,
     257.5, 335.5, 436.5, 567.5, 738.0, 959.5]
)
N_LOG_BINS = 12
DLN_ELL = 0.4
LOG_EDGES = LMAX * np.exp(-N_LOG_BINS * DLN_ELL) * np.exp(DLN_ELL * np.arange(N_LOG_BINS + 1))


def binary_disc_mask(nside: int, lon: np.ndarray, lat: np.ndarray, radius: np.ndarray) -> np.ndarray:
    mask = np.ones(hp.nside2npix(nside), dtype=np.float64)
    for lo, la, rr in zip(lon, lat, radius):
        mask[hp.query_disc(nside, hp.ang2vec(float(lo), float(la), lonlat=True), float(rr))] = 0.0
    return mask


def decoupled_cl_per_ell(ymap: np.ndarray, mask_apo: np.ndarray, pixwin2: np.ndarray) -> np.ndarray:
    w = mask_apo
    m = ymap - float(np.sum(w * ymap) / np.sum(w))
    field = nmt.NmtField(w, [m], lmax=LMAX)
    bins = nmt.NmtBin.from_lmax_linear(LMAX, nlb=1)
    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(field, field, bins)
    cl = workspace.decouple_cell(nmt.compute_coupled_cell(field, field))[0]
    ell_eff = bins.get_effective_ells().astype(int)
    cl_full = np.full(LMAX + 1, np.nan)
    cl_full[ell_eff] = cl
    return cl_full / pixwin2


def bin_dl_18(ell: np.ndarray, cl: np.ndarray) -> np.ndarray:
    dl = ell * (ell + 1.0) * cl / (2.0 * np.pi)
    out = np.full(18, np.nan)
    for i in range(18):
        inside = (ell >= ELL_MIN[i]) & (ell <= ELL_MAX[i])
        out[i] = np.nanmean(dl[inside])
    return out


def bin_cl_log(ell: np.ndarray, cl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centres = np.sqrt(LOG_EDGES[:-1] * LOG_EDGES[1:])
    out = np.empty(N_LOG_BINS)
    for i, (lo, hi) in enumerate(zip(LOG_EDGES[:-1], LOG_EDGES[1:])):
        inside = (ell >= lo) & (ell <= hi) if i == N_LOG_BINS - 1 else (ell >= lo) & (ell < hi)
        if not np.any(inside):
            raise ValueError(f"log bin [{lo}, {hi}] is empty")
        out[i] = np.nanmean(cl[inside])
    return centres, centres * (centres + 1.0) * out / (2.0 * np.pi)


def write_bandpowers(path: Path, ell: np.ndarray, dl_1e12: np.ndarray, header: str) -> None:
    np.savetxt(path, np.column_stack([ell, dl_1e12]), fmt="%.6e", header=header)


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ymap = hp.read_map(str(MAP_FILE), dtype=np.float64)
    nside = hp.npix2nside(ymap.size)
    det = np.load(CAT_FILE)
    q = np.asarray(det["q_opt"], dtype=np.float64)
    lon = np.asarray(det["lon"], dtype=np.float64)
    lat = np.asarray(det["lat"], dtype=np.float64)
    t500_rad = np.deg2rad(np.asarray(det["theta_500"], dtype=np.float64) / 60.0)
    print(f"nside={nside}  N_cat={q.size}  mean y={ymap.mean():.3e}", flush=True)

    pixwin2 = hp.pixwin(nside, lmax=LMAX) ** 2
    ell = np.arange(LMAX + 1, dtype=float)
    mask_floor = np.deg2rad(2.0 * FWHM_ARCMIN / 60.0)

    metadata: dict[str, dict] = {}
    for cut, tag in zip(Q_CUTS, CUT_TAGS):
        keep = q > cut
        n_keep = int(keep.sum())
        radius = np.maximum(R_MULT * t500_rad[keep], mask_floor)
        print(f"=== q>{cut:g}: masking {n_keep} ===", flush=True)
        mask_bin = binary_disc_mask(nside, lon[keep], lat[keep], radius)
        f_sky_raw = float(mask_bin.mean())
        mask_apo = nmt.mask_apodization(mask_bin, APOSIZE_DEG, apotype=APOTYPE)
        f_sky_eff = float(np.mean(mask_apo**2))
        del mask_bin
        print(f"  f_sky raw={f_sky_raw:.4f} eff={f_sky_eff:.4f} ({time.time() - t0:.0f}s)", flush=True)

        cl_masked = decoupled_cl_per_ell(ymap, mask_apo, pixwin2)
        del mask_apo
        dl18 = bin_dl_18(ell, cl_masked)
        ell_log, dl12 = bin_cl_log(ell, cl_masked)

        header = (
            f"L1_m9 truth y masked with sciMMF q_opt>{cut:g}; "
            f"r=max(4*theta500, 2x10arcmin), {APOTYPE} {APOSIZE_DEG} deg, "
            "masked monopole subtracted, NaMaster nlb=1, pixwin deconvolved"
        )
        write_bandpowers(
            OUT_DIR / f"Dl_yy_L1_m9_masked_{tag}_scimmf_binned_18.txt",
            ELL_EFF,
            dl18 * 1e12,
            header + "\nell_eff  1e12_D_ell_yy",
        )
        write_bandpowers(
            OUT_DIR / f"Dl_yy_L1_m9_masked_{tag}_scimmf_logbins_dln0p4_lmax10000.txt",
            ell_log,
            dl12 * 1e12,
            header + f"; Delta ln ell={DLN_ELL}; lmax={LMAX}\nell_eff  1e12_D_ell_yy",
        )
        metadata[tag] = {
            "q_cut": cut,
            "n_masked": n_keep,
            "f_sky_raw": f_sky_raw,
            "f_sky_eff": f_sky_eff,
        }
        print(f"  wrote {tag} ({time.time() - t0:.0f}s)", flush=True)

    for tag, entry in metadata.items():
        assert int((q > entry["q_cut"]).sum()) == entry["n_masked"], tag

    meta = {
        "map": str(MAP_FILE),
        "catalogue": str(CAT_FILE),
        "n_catalogue": int(q.size),
        "masking": {
            "radius": "max(4*theta500, 2*FWHM), FWHM=10 arcmin",
            "theta500": "sciMMF catalogue theta_500 (arcmin)",
            "positions": "sciMMF lon/lat (deg)",
            "apodization": f"{APOTYPE} {APOSIZE_DEG} deg",
            "estimator": f"NaMaster MASTER, nlb=1, lmax={LMAX}, no beam, pixwin deconvolved",
        },
        "cuts": metadata,
        "runtime_seconds": time.time() - t0,
    }
    (OUT_DIR / "L1_m9_masked_scimmf_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
