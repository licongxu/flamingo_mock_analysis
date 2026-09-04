#!/usr/bin/env bash
# One-time tidy of figures/ into topic subfolders (safe to re-run).
set -euo pipefail
cd /scratch/scratch-lxu/flamingo_mock_analysis/figures

mkdir -p noise/ffp10 noise/npipe_vs_ffp10 l1_m9 yang26 szifi

# Noise comparison galleries
for f in ffp10_*; do
  [[ -e "$f" ]] || continue
  mv -n "$f" noise/ffp10/ 2>/dev/null || true
done
for f in npipe_vs_ffp10_*; do
  [[ -e "$f" ]] || continue
  mv -n "$f" noise/npipe_vs_ffp10/ 2>/dev/null || true
done

# L1_m9 feedback / ILC error-band figures
for f in l1_m9_*; do
  [[ -e "$f" ]] || continue
  mv -n "$f" l1_m9/ 2>/dev/null || true
done

# Yang26 cross-comparison
for f in l1_prescriptions_vs_yang26_* l1_m9_vs_yang26_*; do
  [[ -e "$f" ]] || continue
  mv -n "$f" yang26/ 2>/dev/null || true
done

# SZiFi cluster-finding figures
for f in szifi_*; do
  [[ -e "$f" ]] || continue
  mv -n "$f" szifi/ 2>/dev/null || true
done

# HILC: prescription/deproj layout
mkdir -p hilc/combined/nodeproj hilc/L1_m9/legacy_l1m9_scripts

# Flat per-prescription PNGs -> nodeproj/
for p in L1_m9 fgas-8sigma Mstar-1sigma LS8; do
  [[ -d "hilc/$p" ]] || continue
  if [[ -f "hilc/$p/r1xr2_fig9_fullsky.png" && ! -d "hilc/$p/nodeproj" ]]; then
    mkdir -p "hilc/$p/nodeproj"
    for f in hilc/$p/*.png; do
      [[ -f "$f" ]] || continue
      mv -n "$f" "hilc/$p/nodeproj/"
    done
  fi
done

# Legacy L1_m9-only deproj-suite plots
if [[ -d hilc/l1_m9_deproj_suite ]]; then
  mv -n hilc/l1_m9_deproj_suite/* hilc/L1_m9/legacy_l1m9_scripts/ 2>/dev/null || true
  rmdir hilc/l1_m9_deproj_suite 2>/dev/null || true
fi

# Combined overlays at hilc root -> combined/nodeproj/
for f in hilc/r1xr2_*.png; do
  [[ -f "$f" ]] || continue
  mv -n "$f" hilc/combined/nodeproj/
done

echo "figures/ layout:"
find . -maxdepth 3 -type d | sort
