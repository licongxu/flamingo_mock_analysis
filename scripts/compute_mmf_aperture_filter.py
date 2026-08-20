#!/usr/bin/env python3
"""Rebuild y_t(ell), N^{-1}(ell), d(ell) on one tile and form the MMF aperture.

Last noise-curve job saved only sigma_y0(theta), not the harmonic ingredients.
This reconstructs them from the existing tmap/mask npy products.

Aperture from the convolution theorem:
    Y_ap(0) = int d^2ell/(2pi)^2  y(ell) W(ell)
Matching the MMF contraction y_t^dagger N^{-1} d requires
    W(ell) = [y_t^dagger N^{-1} d] / y(ell)
when y(ell) is a Compton-y cutout. Also saves the data-independent
filter F = N^{-1} y_t and the radial average of |W|.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("SZIFI_ARRAY_BACKEND", "numpy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import matplotlib.pyplot as plt
import numpy as np

from flamingo_mock.szifi.paths import SZiFiPaths, TILE_NX
from flamingo_mock.szifi.run import default_params
from flamingo_mock.szifi.tiles import select_pilot_tile_ids

THETA_ARCMIN = 5.0
YMAP = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc/inputs_nside2048_npipe/"
    "compton_y_nside2048.fits"
)


def radial_mean(ell: np.ndarray, val: np.ndarray, n_bins: int = 40) -> tuple[np.ndarray, np.ndarray]:
    ell = np.asarray(ell).ravel()
    val = np.asarray(val).ravel()
    ok = np.isfinite(ell) & np.isfinite(val) & (ell > 1.0)
    ell, val = ell[ok], val[ok]
    bins = np.exp(np.linspace(np.log(ell.min()), np.log(ell.max()), n_bins + 1))
    centres = np.sqrt(bins[:-1] * bins[1:])
    idx = np.digitize(ell, bins) - 1
    out = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = idx == i
        if m.any():
            out[i] = np.mean(val[m])
    return centres, out


def main() -> None:
    import szifi
    from szifi import maps, mmf, model, spec, utils

    paths = SZiFiPaths()
    field_id = select_pilot_tile_ids(n=1, b_min_deg=40.0)[0]
    split = "A"
    print(f"tile={field_id} split={split} theta_500={THETA_ARCMIN}'")

    tmap_path = paths.tmap_path(split, field_id)
    mask_path = paths.mask_path(split, field_id)
    assert tmap_path.is_file(), tmap_path
    assert mask_path.is_file(), mask_path
    print(f"have tmap {tmap_path.name}  mask {mask_path.name}")

    params_szifi, params_data, params_model = default_params(
        paths, [field_id], split=split, mmf_type="standard"
    )
    params_szifi["iterative"] = False
    params_szifi["inpaint"] = False
    params_szifi["array_backend"] = "numpy"

    data = szifi.input_data(params_szifi=params_szifi, params_data=params_data)
    d = data.data
    exp = d["experiment"]
    cosmology = model.cosmological_model(params_szifi).cosmology
    dx = d["dx_arcmin"][field_id] / 60.0 / 180.0 * np.pi
    pix = maps.pixel(TILE_NX, dx)
    t_obs = np.asarray(d["t_obs"][field_id], dtype=np.float32)
    t_noi = np.asarray(d["t_noi"][field_id], dtype=np.float32)
    mask_ps = np.asarray(d["mask_ps"][field_id], dtype=np.float64)
    mask_point = np.asarray(d["mask_point"][field_id], dtype=np.float64)

    if params_szifi["a_matrix"] is None:
        a_matrix = np.zeros((len(exp.tsz_sed), 1))
        a_matrix[:, 0] = exp.tsz_sed
        params_szifi["a_matrix"] = a_matrix

    ps = spec.power_spectrum(
        pix, mask=mask_ps, cm_compute=False, cm_compute_scratch=False,
        cm_save=False, cm_name=None, bin_fac=params_szifi["powspec_bin_fac"],
    )
    cspec = spec.cross_spec(np.arange(len(params_szifi["freqs"])))
    cspec.get_cross_spec(
        pix, t_map=t_noi, ps=ps, decouple_type=params_szifi["decouple_type"],
        inpaint_flag=False, mask_point=mask_point, lsep=params_szifi["lsep"],
        bin_fac=params_szifi["powspec_bin_fac"],
    )
    inv_cov = cspec.get_inv_cov(
        pix, t_map=t_noi, interp_type=params_szifi["interp_type"],
        bin_fac=params_szifi["powspec_bin_fac"], mask=mask_ps,
        cov_type=params_szifi["cov_type"],
        cov_kernel_shape=params_szifi["cov_kernel_shape"],
    )
    params_szifi["cmmf_type"] = "standard_mmf"
    cmmf = mmf.scmmf_precomputation(
        pix=pix, freqs=params_szifi["freqs"], inv_cov=inv_cov,
        lrange=params_szifi["lrange"], beam_type=params_szifi["beam"],
        exp=exp, cmmf_type=params_szifi["cmmf_type"],
        a_matrix=params_szifi["a_matrix"],
        comp_to_calculate=params_szifi["comp_to_calculate"],
        mmf_type="standard",
    )

    z = 0.2
    m500 = model.get_m_500(THETA_ARCMIN, z, cosmology)
    nfw = model.gnfw(m500, z, cosmology, type=params_model["profile_type"])
    theta_cart = [(0.5 * pix.nx) * pix.dx, (0.5 * pix.nx) * pix.dx]
    t_tem = nfw.get_t_map_convolved(
        pix, exp, beam=params_szifi["beam"], theta_cart=theta_cart,
        get_nc=False, sed=False,
    )
    t_tem = t_tem / nfw.get_y_norm(params_szifi["norm_type"])
    tem = maps.filter_tmap(t_tem, pix, params_szifi["lrange"])

    # Harmonic ingredients (SZiFi FFT convention).
    yt_fft = maps.get_fft_f(tem, pix)
    d_fft = maps.get_fft_f(t_obs, pix)
    yt_sed = maps.get_tmap_times_fvec(yt_fft, cmmf.a_matrix[:, 0])
    yt_sed_cut = maps.reshape_ell_matrix(yt_sed, inv_cov.shape[:2])
    d_fft_cut = maps.reshape_ell_matrix(d_fft, inv_cov.shape[:2])
    # F = N^{-1} y_t   shape (nell, nell, nfreq)
    F = utils.get_inv_cov_conjugate(yt_sed_cut, inv_cov)
    # S(ell) = y_t^dagger N^{-1} d  (complex scalar per mode)
    S = np.einsum("ijk,ijk->ij", np.conjugate(yt_sed_cut), utils.get_inv_cov_conjugate(d_fft_cut, inv_cov))
    N_yt = np.einsum("ijk,ijk->ij", np.conjugate(yt_sed_cut), F)

    ell = maps.rmap(pix).get_ell()
    ell_cut = maps.reshape_ell_matrix(ell[..., None], inv_cov.shape[:2])[..., 0]

    # Compton-y cutout for the note formula W = S / y(ell)
    W = None
    y_fft = None
    if YMAP.is_file():
        import healpy as hp
        from flamingo_mock.szifi.paths import TILE_L_DEG, TILE_NSIDE
        from szifi.sphere import get_cutout

        y_hp = np.asarray(hp.read_map(str(YMAP), dtype=np.float64))
        lon, lat = hp.pix2ang(TILE_NSIDE, field_id, lonlat=True)
        y_cut = np.asarray(get_cutout(y_hp, [lon, lat], TILE_NX, TILE_L_DEG), dtype=np.float64)
        y_fft_full = maps.get_fft(y_cut, pix)
        y_fft = maps.reshape_ell_matrix(y_fft_full[..., None], inv_cov.shape[:2])[..., 0]
        W = np.where(np.abs(y_fft) > 1e-20, S / y_fft, np.nan)
        print(f"Compton-y cutout from {YMAP.name}: y_fft {y_fft.shape}")
    else:
        print(f"no Compton-y map at {YMAP}; saving S(ell) only")

    print("shapes:")
    print(f"  d_fft      {d_fft.shape}")
    print(f"  y_t(ell)   {yt_sed.shape}")
    print(f"  N^{-1}     {inv_cov.shape}")
    print(f"  F=N^{-1}y_t {F.shape}")
    print(f"  S=y_t^H N^{-1} d {S.shape}")

    out_dir = paths.catalogues_dir() / "mmf_aperture"
    out_dir.mkdir(parents=True, exist_ok=True)
    save = {
        "ell": ell_cut.astype(np.float32),
        "yt_fft": yt_sed_cut.astype(np.complex64),
        "d_fft": d_fft_cut.astype(np.complex64),
        "inv_cov": inv_cov.astype(np.complex64),
        "F_Ninv_yt": F.astype(np.complex64),
        "S_ytH_Ninv_d": S.astype(np.complex64),
        "N_yt": N_yt.astype(np.complex64),
        "field_id": field_id,
        "theta_500_arcmin": THETA_ARCMIN,
        "split": split,
    }
    if y_fft is not None:
        save["y_fft"] = y_fft.astype(np.complex64)
        save["W_aperture"] = W.astype(np.complex64)
    np.savez_compressed(out_dir / f"field_{field_id}_theta{THETA_ARCMIN:.0f}.npz", **save)
    print(f"Wrote {out_dir / f'field_{field_id}_theta{THETA_ARCMIN:.0f}.npz'}")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    ell_r, s_r = radial_mean(ell_cut, np.abs(S))
    axes[0].loglog(ell_r, s_r, "k-")
    axes[0].set_xlabel(r"$\ell$")
    axes[0].set_ylabel(r"$|y_t^\dagger N^{-1} d|(\ell)$")
    axes[0].set_title(r"MMF integrand $S(\ell)$")
    axes[0].grid(True, which="both", alpha=0.3)

    ell_r, n_r = radial_mean(ell_cut, np.abs(N_yt))
    axes[1].loglog(ell_r, n_r, "C0-")
    axes[1].set_xlabel(r"$\ell$")
    axes[1].set_ylabel(r"$|y_t^\dagger N^{-1} y_t|(\ell)$")
    axes[1].set_title(r"Filter normalisation kernel")
    axes[1].grid(True, which="both", alpha=0.3)

    if W is not None:
        ell_r, w_r = radial_mean(ell_cut, np.abs(W))
        axes[2].loglog(ell_r, w_r, "C1-")
        axes[2].set_ylabel(r"$|W(\ell)|$")
        axes[2].set_title(r"$W=S/y$ (aperture)")
    else:
        # Fallback: isotropic filter amplitude from F at 143 GHz (channel 1)
        ell_r, f_r = radial_mean(ell_cut, np.abs(F[:, :, 1]))
        axes[2].loglog(ell_r, f_r, "C1-")
        axes[2].set_ylabel(r"$|[N^{-1}y_t]_{143}|$")
        axes[2].set_title("MMF filter (143 GHz)")
    axes[2].set_xlabel(r"$\ell$")
    axes[2].grid(True, which="both", alpha=0.3)
    fig.suptitle(
        rf"Tile {field_id}, $\theta_{{500}}={THETA_ARCMIN}'$, non-iterative iMMF",
        y=1.03,
    )
    fig.tight_layout()
    fig_path = Path("figures/mmf_aperture_filter_field0")
    fig.savefig(fig_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {fig_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
