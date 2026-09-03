#!/usr/bin/env python3
"""Run missing HILC y-maps for the full prescription × deprojection grid."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from hilc_prescriptions import (  # noqa: E402
    ALL_DEPROJ,
    ALL_RUNS,
    hilc_ymap,
    write_hilc_yaml,
    yaml_path,
)

RUN = _SCRIPTS / "run_hilc.py"


def main() -> None:
    todo: list[Path] = []
    for name in ALL_RUNS:
        for deproj in ALL_DEPROJ:
            for masked in (False, True):
                for real in (1, 2):
                    write_hilc_yaml(name, masked=masked, real=real, deproj=deproj)
                    yp = hilc_ymap(name, masked=masked, real=real, deproj=deproj)
                    cfg = yaml_path(name, masked=masked, real=real, deproj=deproj)
                    if yp.is_file() and yp.stat().st_size > 1_000_000:
                        print(f"skip {name} {deproj.key} {'q5' if masked else 'fs'} r{real}")
                        continue
                    todo.append(cfg)
    print(f"running {len(todo)} HILC configs ...", flush=True)
    for cfg in todo:
        print(f"===== {cfg.name} =====", flush=True)
        subprocess.run([sys.executable, str(RUN), str(cfg)], check=True)


if __name__ == "__main__":
    main()
