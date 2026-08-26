"""Run pyILC HILC on the homog test maps with the q>5 cluster mask applied."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

CFG = Path("/scratch/scratch-lxu/flamingo_mock_analysis/configs/hilc_y_flamingo_homog_q5masked.yml")
ILC_SRC = Path("/scratch/scratch-lxu/flamingo_needlet_ilc/src")


def main() -> None:
    sys.path.insert(0, str(ILC_SRC))
    from flamingo_mock.ilc.run import run_ilc

    t0 = time.time()
    ypath = run_ilc(CFG, backend=os.environ.get("PYILC_BACKEND", "jax"))
    print(f"[driver] y_map={ypath}  elapsed={time.time()-t0:.1f}s")
    y = np.asarray(__import__("healpy").read_map(str(ypath), dtype=np.float64))
    print(f"[driver] y rms={float(y.std()):.4g}  nside={__import__('healpy').get_nside(y)}")


if __name__ == "__main__":
    main()
