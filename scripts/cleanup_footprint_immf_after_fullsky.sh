#!/bin/bash
# After homog full-sky iMMF exits successfully, delete the footprint (masked) catalogue.
# Idle wait only — no extra MMF/CPU.
set -euo pipefail
PARENT="${1:-282952}"
CAT="/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi_homog/catalogues"
FULL="$CAT/homog_immf_fullsky_splitA_immf_q5.npz"
LOG="/scratch/scratch-lxu/flamingo_mock_analysis/logs/cleanup_footprint_immf.log"

{
  echo "$(date -Is) waiting for full-sky parent pid=$PARENT"
  while kill -0 "$PARENT" 2>/dev/null; do
    sleep 60
  done
  while pgrep -f 'flamingo-szifi run --kind homog --full-sky' >/dev/null 2>&1; do
    sleep 30
  done
  echo "$(date -Is) full-sky process gone"
  if [[ -s "$FULL" ]]; then
    rm -f "$CAT/homog_immf_splitA_immf_q5.npz" "$CAT/homog_immf_splitA_immf_q5.json"
    rm -rf "$CAT/partial_homog_immf_splitA_immf"
    echo "$(date -Is) deleted footprint catalogue + partials"
    ls -lh "$FULL"
  else
    echo "$(date -Is) ERROR: $FULL missing or empty — left footprint catalogue in place"
    exit 1
  fi
} >>"$LOG" 2>&1
