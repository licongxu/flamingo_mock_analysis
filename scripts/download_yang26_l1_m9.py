#!/usr/bin/env python3
"""Download Yang et al. (2026) L1_m9 integrated maps from COSMA DataWeb.

Writes HDF5 files under
``/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/L1_m9/{run}/``
without touching the existing L2p8 FITS products in ``components/{cmb,cib,tsz,ksz}/``.

Source: https://dataweb.cosma.dur.ac.uk:8443/flamingo/viewer.html
"""
from __future__ import annotations

import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import hdfstream
import requests

OUT_ROOT = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/components/L1_m9")
DOWNLOAD_BASE = "https://dataweb.cosma.dur.ac.uk:8443/hdfstream/download"
COSMA_PREFIX = "FLAMINGO/L1_m9"

# COSMA run folder -> local folder (fiducial is L1_m9/L1_m9 on the portal).
RUNS = {
    "L1_m9": "L1_m9",
    "fgas-8sigma": "fgas-8sigma",
    "Mstar-1sigma": "Mstar-1sigma",
    "LS8": "LS8",
}

WORKERS = 3
CHUNK = 8 * 1024 * 1024
RETRIES = 4


def remote_listing(run: str) -> list[tuple[str, int]]:
    path = f"{COSMA_PREFIX}/{run}/integrated_maps/yang26/lightcone0_shells"
    d = hdfstream.open("cosma", path)
    files = []
    for name, f in d.files.items():
        files.append((name, int(f.size)))
    files.sort()
    return files


def download_one(run: str, name: str, size: int) -> str:
    dest = OUT_ROOT / run / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size:
        return f"skip  {run}/{name}"

    url = f"{DOWNLOAD_BASE}/{COSMA_PREFIX}/{run}/integrated_maps/yang26/lightcone0_shells/{name}"
    part = dest.with_suffix(dest.suffix + ".part")

    last_err = None
    for attempt in range(1, RETRIES + 1):
        t0 = time.time()
        n = 0
        hasher = hashlib.md5()
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(part, "wb") as out:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        out.write(chunk)
                        hasher.update(chunk)
                        n += len(chunk)
            if n != size:
                raise RuntimeError(f"size mismatch: got {n} expected {size}")
            part.replace(dest)
            dt = time.time() - t0
            mbps = (n / 1e6) / dt if dt > 0 else 0.0
            return f"ok    {run}/{name}  {n/1e9:.2f} GB  {mbps:.0f} MB/s  md5={hasher.hexdigest()[:8]}"
        except Exception as e:
            last_err = e
            print(f"retry {attempt}/{RETRIES} {run}/{name}: {e}", flush=True)
            if part.exists():
                part.unlink()
            time.sleep(min(30, 2 ** attempt))
    return f"FAIL  {run}/{name}  {last_err}"


def main() -> int:
    jobs: list[tuple[str, str, int]] = []
    total = 0
    print(f"out: {OUT_ROOT}", flush=True)
    for run in RUNS:
        files = remote_listing(run)
        nbytes = sum(s for _, s in files)
        print(f"{run}: {len(files)} files, {nbytes/1024**3:.2f} GiB", flush=True)
        for name, size in files:
            print(f"  {name:70s} {size/1024**2:8.1f} MiB", flush=True)
            jobs.append((run, name, size))
            total += size
    print(f"total {len(jobs)} files, {total/1024**3:.2f} GiB, workers={WORKERS}", flush=True)

    n_fail = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(download_one, run, name, size) for run, name, size in jobs]
        for fut in as_completed(futs):
            msg = fut.result()
            print(msg, flush=True)
            if msg.startswith("FAIL"):
                n_fail += 1
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
