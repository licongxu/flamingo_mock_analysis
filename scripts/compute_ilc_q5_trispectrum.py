#!/usr/bin/env python3
"""q>5 one-halo trispectrum covariance at the L1_m9 SZiFi ILC mask f_sky.

Same hmfast pipeline as flamingo_repo compute_l1_m9_customgnfw_bestfit_covariance.py
(A_SZ=-4.1075, q_cat=5, W4). f_sky is ⟨w²⟩ of cluster_mask_apo("L1_m9"), not qfrommap.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "1"))

import healpy as hp
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
_FLAMINGO_SCRIPTS = Path("/scratch/scratch-lxu/flamingo_repo/scripts")
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_FLAMINGO_SCRIPTS))
from hilc_prescriptions import cluster_mask_apo  # noqa: E402

A_SZ = -4.1075073
Q_CAT = 5.0
OUT_DIR = _ROOT / "data"
COV_NAME = "cov_trispectrum_L1_m9_ilc_q5_Dl_yy_binned_18.npy"


def _load_cov_mod():
    spec = importlib.util.spec_from_file_location(
        "l1_m9_cov",
        _FLAMINGO_SCRIPTS / "compute_l1_m9_customgnfw_bestfit_covariance.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ilc_fsky() -> float:
    w = np.asarray(hp.read_map(str(cluster_mask_apo("L1_m9")), dtype=np.float64))
    return float(np.mean(w**2))


def main() -> None:
    f_sky = ilc_fsky()
    mod = _load_cov_mod()
    halo_model, tracer, noise_coeff = mod._theory_context(A_SZ)
    raw = mod._raw_theory(halo_model, tracer, noise_coeff, A_SZ, Q_CAT)
    result = mod.build_covariance_18(raw, f_sky)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cov_path = OUT_DIR / COV_NAME
    np.save(cov_path, result["cov_trispectrum"])
    theory_path = OUT_DIR / "theory_L1_m9_ilc_q5_Dl_yy_binned_18.txt"
    np.savetxt(
        theory_path,
        np.column_stack(
            [result["ell"], result["dl_1h"], result["dl_2h"], result["dl_total"]]
        ),
        header="ell_eff D_ell_1h D_ell_2h D_ell_total",
        fmt="%.16e",
    )
    meta = {
        "A_SZ": A_SZ,
        "q_cat": Q_CAT,
        "f_sky": f_sky,
        "f_sky_source": str(cluster_mask_apo("L1_m9")),
        "qfrommap_f_sky_not_used": 0.8595492784996496,
        "cov_trispectrum": str(cov_path),
        "theory": str(theory_path),
    }
    meta_path = OUT_DIR / "cov_trispectrum_L1_m9_ilc_q5_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
