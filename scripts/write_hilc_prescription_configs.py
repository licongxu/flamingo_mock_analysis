"""Write HILC YAMLs for all prescriptions and deprojection cases."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hilc_prescriptions import ALL_DEPROJ, ALL_RUNS, write_hilc_yaml

if __name__ == "__main__":
    for name in ALL_RUNS:
        for deproj in ALL_DEPROJ:
            for masked in (False, True):
                for real in (1, 2):
                    p = write_hilc_yaml(name, masked=masked, real=real, deproj=deproj)
                    print("wrote", p)
