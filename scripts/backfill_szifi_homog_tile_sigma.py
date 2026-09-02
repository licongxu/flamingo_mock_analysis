#!/usr/bin/env python3
"""Backfill per-tile iMMF σ_y0 for homog L1 prescription SZiFi runs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic")
PRESCRIPTIONS = ("L1_m9", "fgas-8sigma", "Mstar-1sigma", "LS8")
SCRIPT = Path(__file__).resolve().parent / "compute_flamingo_immf_skyavg_noise.py"


def main() -> None:
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    for prescription in PRESCRIPTIONS:
        out_root = ROOT / "szifi_homog" / prescription
        total_maps = ROOT / "total_maps" / prescription
        log = Path(__file__).resolve().parents[1] / "logs" / f"sigma_per_tile_{prescription}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--kind",
            "homog",
            "--iterative",
            "--runner-cache",
            "--no-plot",
            "--full-sky",
            "--n-workers",
            str(n_workers),
            "--threads-per-worker",
            "1",
            "--out-root",
            str(out_root),
            "--total-maps-dir",
            str(total_maps),
        ]
        print(f"=== {prescription} ===", flush=True)
        with log.open("w") as fh:
            fh.write(" ".join(cmd) + "\n\n")
            fh.flush()
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
        if proc.returncode != 0:
            raise SystemExit(f"{prescription} failed; see {log}")
        print(f"done {prescription} → {log}", flush=True)


if __name__ == "__main__":
    main()
