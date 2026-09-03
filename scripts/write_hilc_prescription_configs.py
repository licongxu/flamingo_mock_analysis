"""Write no-deproj HILC YAMLs for fgas-8sigma, Mstar-1sigma, LS8."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hilc_prescriptions import PRESCRIPTIONS, write_hilc_yaml

if __name__ == "__main__":
    for name in PRESCRIPTIONS:
        for masked in (False, True):
            for real in (1, 2):
                p = write_hilc_yaml(name, masked=masked, real=real)
                print("wrote", p)
