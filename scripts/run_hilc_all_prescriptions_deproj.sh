#!/usr/bin/env bash
# HILC: all prescriptions × deprojection × (full sky | q>5 masked) × (r1 | r2).
# Skips y-maps that already exist. Then replots.
set -euo pipefail
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYILC_BACKEND=jax
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

ROOT=/scratch/scratch-lxu/flamingo_mock_analysis
cd "$ROOT"
mkdir -p logs
LOG=logs/hilc_all_prescriptions_deproj.log
echo "===== HILC all prescriptions + deproj $(date -Iseconds) =====" | tee "$LOG"

python scripts/write_hilc_prescription_configs.py 2>&1 | tee -a "$LOG"
python scripts/run_hilc_prescriptions.py 2>&1 | tee -a "$LOG"
python scripts/plot_hilc_homog_prescriptions.py 2>&1 | tee -a "$LOG"
echo "===== ALL DONE $(date -Iseconds) =====" | tee -a "$LOG"
