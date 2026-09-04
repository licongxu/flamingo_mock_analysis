#!/usr/bin/env bash
# HILC (no deproj) on fgas-8sigma, Mstar-1sigma, LS8: full sky + q>5-masked, r1 and r2.
set -euo pipefail
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYILC_BACKEND=jax
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

ROOT=/scratch/scratch-lxu/flamingo_mock_analysis
cd "$ROOT"
mkdir -p logs
LOG=logs/hilc_feedback_prescriptions.log
echo "===== HILC prescriptions $(date -Iseconds) =====" | tee "$LOG"

for p in fgas-8sigma Mstar-1sigma LS8; do
  for cfg in \
    configs/hilc_y_flamingo_homog_${p}.yml \
    configs/hilc_y_flamingo_homog_${p}_r2.yml \
    configs/hilc_y_flamingo_homog_${p}_q5masked.yml \
    configs/hilc_y_flamingo_homog_${p}_q5masked_r2.yml
  do
    echo "===== RUN $cfg $(date -Iseconds) =====" | tee -a "$LOG"
    python scripts/run_hilc.py "$cfg" 2>&1 | tee -a "$LOG"
  done
done

echo "===== PLOT $(date -Iseconds) =====" | tee -a "$LOG"
python scripts/plot_hilc_homog_prescriptions.py 2>&1 | tee -a "$LOG"
echo "===== ALL DONE $(date -Iseconds) =====" | tee -a "$LOG"
