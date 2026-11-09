#!/usr/bin/env python3
"""HILC follow-up (hybrid, fiducial-mask, CIB-shuffle). CPU ducc + numpy ILC.

Does not overwrite existing M*/fgas q>5 outputs: those jobs use extra_tag=fidmask.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from hilc_prescriptions import (  # noqa: E402
    CIBSHUFFLE,
    DEPROJ_CIB_DBETA,
    DEPROJ_FOLLOWUP,
    DEPROJ_NONE,
    FIDMASK_TAG,
    HYBRID_FID_TSZ_MSTAR_CIB,
    HYBRID_MSTAR_TSZ_FID_CIB,
    hilc_ymap,
    write_hilc_yaml,
)

RUN = _SCRIPTS / "run_hilc.py"


def jobs():
    out = []
    for name in (HYBRID_FID_TSZ_MSTAR_CIB, HYBRID_MSTAR_TSZ_FID_CIB):
        for deproj in DEPROJ_FOLLOWUP:
            for real in (1, 2):
                out.append((name, deproj, real, "L1_m9", ""))
    for name in ("Mstar-1sigma", "fgas-8sigma"):
        for deproj in (DEPROJ_NONE, DEPROJ_CIB_DBETA):
            for real in (1, 2):
                out.append((name, deproj, real, "L1_m9", FIDMASK_TAG))
    for deproj in (DEPROJ_NONE, DEPROJ_CIB_DBETA):
        for real in (1, 2):
            out.append((CIBSHUFFLE, deproj, real, "L1_m9", ""))
    return out


def main() -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYILC_BACKEND"] = "numpy"
    todo = []
    for name, deproj, real, mask_name, extra_tag in jobs():
        cfg = write_hilc_yaml(
            name,
            masked=True,
            real=real,
            deproj=deproj,
            mask_name=mask_name,
            extra_tag=extra_tag,
        )
        yp = hilc_ymap(
            name, masked=True, real=real, deproj=deproj, extra_tag=extra_tag
        )
        if yp.is_file() and yp.stat().st_size > 1_000_000:
            print(f"skip {yp.name}")
            continue
        todo.append(cfg)
    print(f"running {len(todo)} HILC configs (numpy + ducc CPU) ...", flush=True)
    for cfg in todo:
        print(f"===== {cfg.name} =====", flush=True)
        subprocess.run(
            [sys.executable, str(RUN), str(cfg)], check=True, env=env
        )


if __name__ == "__main__":
    main()
