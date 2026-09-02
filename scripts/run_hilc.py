"""Run pyILC HILC from one or more YAML configs.

Usage:
  python scripts/run_hilc.py configs/hilc_y_flamingo_homog.yml
  python scripts/run_hilc.py configs/hilc_y_flamingo_homog*.yml
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "4")

ILC_SRC = Path("/scratch/scratch-lxu/flamingo_needlet_ilc/src")


def run_one(cfg: Path) -> None:
    sys.path.insert(0, str(ILC_SRC))
    from flamingo_mock.ilc.run import run_ilc

    t0 = time.time()
    ypath = run_ilc(cfg, backend=os.environ.get("PYILC_BACKEND", "jax"))
    print(f"[driver] y_map={ypath}  elapsed={time.time()-t0:.1f}s")
    y = np.asarray(__import__("healpy").read_map(str(ypath), dtype=np.float64))
    print(f"[driver] y rms={float(y.std()):.4g}  nside={__import__('healpy').get_nside(y)}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python scripts/run_hilc.py CONFIG.yml [...]")
    for cfg in (Path(a).resolve() for a in sys.argv[1:]):
        if not cfg.is_file():
            raise FileNotFoundError(cfg)
        print(f"===== {cfg.name} =====", flush=True)
        run_one(cfg)


if __name__ == "__main__":
    main()
