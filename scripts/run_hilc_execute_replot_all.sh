#!/usr/bin/env bash
# Re-run homog HILC configs only (TopHatHarmonic; no NILC), overwrite y-maps, replot.
set -euo pipefail

source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYILC_BACKEND=jax
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

ROOT=/scratch/scratch-lxu/flamingo_mock_analysis
cd "$ROOT"
mkdir -p logs
LOG=logs/hilc_execute_replot_all.log
echo "===== HILC execute + replot $(date -Iseconds) =====" | tee "$LOG"

for cfg in \
  configs/hilc_y_flamingo_homog.yml \
  configs/hilc_y_flamingo_homog_r2.yml \
  configs/hilc_y_flamingo_homog_q5masked.yml \
  configs/hilc_y_flamingo_homog_q5masked_r2.yml \
  configs/hilc_y_flamingo_homog_deproj_cib.yml \
  configs/hilc_y_flamingo_homog_r2_deproj_cib.yml \
  configs/hilc_y_flamingo_homog_q5masked_deproj_cib.yml \
  configs/hilc_y_flamingo_homog_q5masked_r2_deproj_cib.yml \
  configs/hilc_y_flamingo_homog_deproj_cib_moments.yml \
  configs/hilc_y_flamingo_homog_r2_deproj_cib_moments.yml \
  configs/hilc_y_flamingo_homog_q5masked_deproj_cib_moments.yml \
  configs/hilc_y_flamingo_homog_q5masked_r2_deproj_cib_moments.yml
do
  echo "===== RUN $cfg $(date -Iseconds) =====" | tee -a "$LOG"
  python scripts/run_hilc.py "$cfg" 2>&1 | tee -a "$LOG"
done

export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
for s in \
  scripts/plot_hilc_homog_r1xr2_split_diagnostics.py \
  scripts/plot_hilc_homog_auto_residuals.py \
  scripts/plot_hilc_homog_auto_fig9.py \
  scripts/plot_hilc_homog_r1xr2_fig9.py \
  scripts/plot_hilc_homog_q5masked_r1xr2_fig9.py \
  scripts/plot_hilc_homog_q5masked.py \
  scripts/plot_hilc_homog_pyilc_convention.py \
  scripts/plot_hilc_weights_fullsky_vs_q5masked.py
do
  echo "===== PLOT $s $(date -Iseconds) =====" | tee -a "$LOG"
  python "$s" 2>&1 | tee -a "$LOG"
done

echo "===== ALL DONE $(date -Iseconds) =====" | tee -a "$LOG"
ls -lt figures/hilc_homog*.png | head -25 | tee -a "$LOG"
