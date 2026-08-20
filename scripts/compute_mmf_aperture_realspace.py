#!/usr/bin/env python3
"""Smooth W(theta/theta_500) vs top-hat at theta_500 = 2', 5', 10'.

Reuse saved tile-0 inv_cov, d, y. Rebuild only y_t(theta_500).
Radial profile: cubic interpolation on a polar ring (continuous curve).
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("SZIFI_ARRAY_BACKEND", "numpy")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

from flamingo_mock.szifi.paths import TILE_L_DEG, TILE_NX, SZiFiPaths
from flamingo_mock.szifi.run import default_params

HARMONIC = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues/"
    "mmf_aperture/field_0_theta5.npz"
)
THETAS = (2.0, 5.0, 10.0)
N_R = 401
N_PHI = 360


def polar_ring(w2d: np.ndarray, dx: float, r_arcmin: np.ndarray) -> np.ndarray:
    """Azimuthal mean of W at radii r_arcmin, cubic interpolation."""
    nx = w2d.shape[0]
    cx = nx // 2
    r_pix = (r_arcmin / (180.0 * 60.0 / np.pi)) / dx
    phi = np.linspace(0.0, 2.0 * np.pi, N_PHI, endpoint=False)
    cphi, sphi = np.cos(phi), np.sin(phi)
    out = np.empty(r_arcmin.size, dtype=np.float64)
    out[0] = float(w2d[cx, cx])
    for i in range(1, r_arcmin.size):
        col = r_pix[i] * cphi + cx
        row = r_pix[i] * sphi + cx
        out[i] = float(
            map_coordinates(w2d, np.vstack([row, col]), order=3, mode="nearest").mean()
        )
    return out


def main() -> None:
    import szifi
    from szifi import maps, model, utils

    saved = np.load(HARMONIC)
    inv_cov = np.asarray(saved["inv_cov"], dtype=np.complex128)
    d_fft = np.asarray(saved["d_fft"], dtype=np.complex128)
    y_fft = np.asarray(saved["y_fft"], dtype=np.complex128)
    field_id = int(saved["field_id"])

    paths = SZiFiPaths()
    params_szifi, params_data, params_model = default_params(
        paths, [field_id], split="A"
    )
    params_szifi["iterative"] = False
    params_szifi["inpaint"] = False
    params_szifi["array_backend"] = "numpy"
    data = szifi.input_data(params_szifi=params_szifi, params_data=params_data)
    exp = data.data["experiment"]
    if params_szifi["a_matrix"] is None:
        a_matrix = np.zeros((len(exp.tsz_sed), 1))
        a_matrix[:, 0] = exp.tsz_sed
        params_szifi["a_matrix"] = a_matrix
    cosmology = model.cosmological_model(params_szifi).cosmology
    dx = TILE_L_DEG / TILE_NX / 180.0 * np.pi
    pix = maps.pixel(TILE_NX, dx)
    theta_cart = [(0.5 * pix.nx) * pix.dx, (0.5 * pix.nx) * pix.dx]
    z = 0.2
    te = np.linspace(0.0, 5.0, N_R)
    colours = {2.0: "C0", 5.0: "C1", 10.0: "C3"}

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot([0, 1, 1, 5], [1, 1, 0, 0], color="0.25", ls="--", lw=1.6,
            label=r"top-hat $\theta<\theta_{500}$")

    profiles = {}
    for th in THETAS:
        nfw = model.gnfw(model.get_m_500(th, z, cosmology), z, cosmology,
                         type=params_model["profile_type"])
        t_tem = nfw.get_t_map_convolved(
            pix, exp, beam=params_szifi["beam"], theta_cart=theta_cart,
            get_nc=False, sed=False,
        )
        t_tem = t_tem / nfw.get_y_norm(params_szifi["norm_type"])
        tem = maps.filter_tmap(t_tem, pix, params_szifi["lrange"])
        yt = maps.reshape_ell_matrix(
            maps.get_tmap_times_fvec(maps.get_fft_f(tem, pix), params_szifi["a_matrix"][:, 0]),
            inv_cov.shape[:2],
        )
        S = np.einsum(
            "ijk,ijk->ij",
            np.conjugate(yt),
            utils.get_inv_cov_conjugate(d_fft, inv_cov),
        )
        W_ell = np.divide(S, y_fft, out=np.zeros_like(S), where=np.abs(y_fft) > 1e-12)
        W_ell = np.where(np.isfinite(W_ell), W_ell, 0.0)
        w2d = np.asarray(maps.get_ifft(W_ell, pix).real, dtype=np.float64)
        w_r = polar_ring(w2d, dx, te * th)
        w_n = w_r / w_r[0]
        profiles[th] = w_n
        ax.plot(te, w_n, color=colours[th], lw=2.0, label=rf"$\theta_{{500}}={th:.0f}'$")
        print(f"theta_500={th}'  W(1)/W(0)={np.interp(1.0, te, w_n):.4f}")

    w2, w5, w10 = profiles[2.0], profiles[5.0], profiles[10.0]
    print(
        f"max |W_2-W_5|={np.nanmax(np.abs(w2-w5)):.4f}  "
        f"|W_5-W_10|={np.nanmax(np.abs(w5-w10)):.4f}"
    )

    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_xlabel(r"$\theta/\theta_{500}$")
    ax.set_ylabel(r"$W(\theta)/W(0)$")
    ax.set_title(r"$W(\theta/\theta_{500})$ vs top-hat")
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.4, 1.15)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = Path("figures/mmf_W_theta_vs_tophat")
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    np.savez(out.with_suffix(".npz"), theta_eff=te, **{f"W_{t:.0f}": profiles[t] for t in THETAS})
    print("Wrote", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
