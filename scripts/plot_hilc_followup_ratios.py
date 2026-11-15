#!/usr/bin/env python3
"""HILC follow-up ratios: hybrid skies, fiducial mask, CIB shuffle."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from pub_style import apply_pub_style, savefig
from hilc_prescriptions import (
    CIBSHUFFLE,
    FIDMASK_TAG,
    HYBRID_FID_TSZ_MSTAR_CIB,
    HYBRID_MSTAR_TSZ_FID_CIB,
    ILC,
)

ROOT = Path(__file__).resolve().parents[1]
PS = ILC / "ps_namaster"
FIG_DIR = ROOT / "figures" / "l1_m9"
STEM = "l1_m9_hilc_followup_ratios"


def _p18(stem: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(PS / f"{stem}_p18.npz")
    return np.asarray(z["ell"]), np.asarray(z["dl"])


def main() -> None:
    apply_pub_style()
    ell, dl_fid_none = _p18("L1_m9_nodeproj_masked")
    _, dl_fid_cib = _p18("L1_m9_deproj_cib_masked")
    _, dl_fid_db = _p18("L1_m9_deproj_cib_dbeta_masked")
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), sharex=True, sharey=True, layout="constrained")
    panels = (
        (
            "hybrid / fiducial mask",
            [
                (f"{HYBRID_FID_TSZ_MSTAR_CIB}_nodeproj_masked", dl_fid_none, "#1a1a1a", "o", "-", r"fid tSZ+$M_\ast$ CIB, no deproj."),
                (f"{HYBRID_FID_TSZ_MSTAR_CIB}_deproj_cib_masked", dl_fid_cib, "#C44E00", "s", "--", r"fid tSZ+$M_\ast$ CIB, CIB"),
                (f"{HYBRID_FID_TSZ_MSTAR_CIB}_deproj_cib_dbeta_masked", dl_fid_db, "#6A1B9A", "^", "--", r"fid tSZ+$M_\ast$ CIB, $\delta\beta$"),
                (f"{HYBRID_MSTAR_TSZ_FID_CIB}_nodeproj_masked", dl_fid_none, "#0055A4", "o", ":", r"$M_\ast$ tSZ+fid CIB, no deproj."),
                (f"{HYBRID_MSTAR_TSZ_FID_CIB}_deproj_cib_dbeta_masked", dl_fid_db, "#00796B", "v", "-.", r"$M_\ast$ tSZ+fid CIB, $\delta\beta$"),
            ],
        ),
        (
            r"own sky, fiducial $q>5$ mask",
            [
                (f"Mstar-1sigma_nodeproj_masked_{FIDMASK_TAG}", dl_fid_none, "#0055A4", "o", "-", r"$M_\ast$, no deproj."),
                (f"Mstar-1sigma_deproj_cib_dbeta_masked_{FIDMASK_TAG}", dl_fid_db, "#6A1B9A", "^", "--", r"$M_\ast$, $\delta\beta$"),
                (f"fgas-8sigma_nodeproj_masked_{FIDMASK_TAG}", dl_fid_none, "#1a1a1a", "s", ":", r"$f_{\mathrm{gas}}$, no deproj."),
                (f"fgas-8sigma_deproj_cib_dbeta_masked_{FIDMASK_TAG}", dl_fid_db, "#C44E00", "D", "-.", r"$f_{\mathrm{gas}}$, $\delta\beta$"),
            ],
        ),
        (
            "CIB shuffle",
            [
                (f"{CIBSHUFFLE}_nodeproj_masked", dl_fid_none, "#0055A4", "o", "-", r"shuffle, no deproj."),
                (f"{CIBSHUFFLE}_deproj_cib_dbeta_masked", dl_fid_db, "#6A1B9A", "^", "--", r"shuffle, $\delta\beta$"),
            ],
        ),
    )
    for ax, (title, curves) in zip(axes, panels):
        ax.axhline(1.0, color="k", lw=1.0)
        for stem, den, color, mk, ls, lab in curves:
            path = PS / f"{stem}_p18.npz"
            if not path.is_file():
                print("missing", path)
                continue
            e, d = _p18(stem)
            ax.semilogx(e, d / den, color=color, marker=mk, ms=4, lw=1.6, ls=ls, label=lab)
        ax.set_xlim(10.0, 959.5)
        ax.set_ylim(0.55, 1.35)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"$\ell$")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(False)
    axes[0].set_ylabel(r"$D_\ell / D_\ell^{\mathrm{fid}}$")
    savefig(fig, STEM, fig_dir=FIG_DIR)
    plt.close(fig)


if __name__ == "__main__":
    main()
