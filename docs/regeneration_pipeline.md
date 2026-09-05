# Homog pipeline regeneration (post z_eff=1.9)

After updating CIB at `z_eff=1.90` (`components/cib/*/CIB_deltaT_{100,143}GHz_*`) and
`total_maps/{L1_m9,fgas-8sigma,Mstar-1sigma,LS8}/`, rebuild downstream products in order.

**Archived (2026-09-05):**

- `…/ilc/archive/2026-09-05_pre_regen/` — old HILC inputs/outputs
- `…/szifi/archive/2026-09-05_pre_regen/` — old NPIPE footprint SZiFi (not used for paper)
- `…/szifi_homog/archive/2026-09-05_pre_regen/` — old homog MMF catalogues/tiles
- `figures/archive/demo/2026-09-05_pre_regen/` — demo figures

Paper science uses **homog white-noise skies only** (`total_maps/` → `szifi_homog/` → HILC).

## Environment

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
cd /scratch/scratch-lxu/flamingo_mock_analysis
export SZIFI_ARRAY_BACKEND=jax   # optional GPU MMF
export PYILC_BACKEND=jax          # HILC on GPU
export CUDA_VISIBLE_DEVICES=1
```

## Step 0 — verify upstream maps (already done)

- `components/cib/{rx}/CIB_deltaT_{100,143}GHz_*` → 2026-09-05, `METHOD` mentions `z_eff=1.9`
- `total_maps/{rx}/sky_*_{100,143}GHz_*` → 2026-09-05
- `217–857 GHz` bands unchanged (Aug 2026) — expected

Optional spectra cache refresh:

```bash
python scripts/check_l1_m9_yang26_cross.py
```

## Step 1 — ILC input skies (K_CMB, nside=2048)

From beamed `total_maps/{prescription}/` + shared r1/r2 white noise:

```bash
for rx in L1_m9 fgas-8sigma Mstar-1sigma LS8; do
  python scripts/build_homog_r2_test_maps.py --prescription "$rx"
done
```

Writes `ilc/inputs_nside2048_homog*` (r1 observation + r2 independent noise realisation).

## Step 2 — HILC y-maps (all prescriptions × deproj × mask × r1/r2)

Regenerates YAMLs, runs missing pyILC jobs, plots prescription suite:

```bash
bash scripts/run_hilc_all_prescriptions_deproj.sh
# log: logs/hilc_all_prescriptions_deproj.log
```

Long-running (~82 configs). Skips y-maps that already exist and are >1 MB.

## Step 3 — SZiFi homog MMF (szifi_jax, GPU 1)

Detection uses `theta_500` = 25 log-spaced points from 0.5′ to 32′
(`src/flamingo_mock/szifi/run.py` `default_params`). Per-tile MMF noise
`sigma_y0(θ)` is written during each run to
`catalogues/sigma_per_tile_{immf|scimmf}_splitA/`.

```bash
ROOT=/rds/rds-lxu/flamingo/integrated_maps_synthetic
ARCH=$ROOT/szifi_homog/archive/2026-09-05_pre_regen
export CUDA_VISIBLE_DEVICES=""

# Tiles (unmasked GAL=PS=1)
for RX in L1_m9 L1_m9_cibshuffle; do
  flamingo-szifi prepare --kind homog --full-sky --split A --n-workers 6 \
    --out-root "$ROOT/szifi_homog/$RX" \
    --total-maps-dir "$ROOT/total_maps/$RX"
done

# Four GPU runs on CUDA:1 (script pins GPU 1). --no-ref: N(q) will differ vs archive (new CIB + 25-pt grid).
python scripts/run_szifi_jax_homog_immf.py --prescription L1_m9 --mmf-type standard \
  --ref $ARCH/L1_m9/catalogues/szifi_jax_splitA_immf_q5.npz --no-ref
python scripts/run_szifi_jax_homog_immf.py --prescription L1_m9 --mmf-type spectrally_constrained \
  --ref $ARCH/L1_m9/catalogues/szifi_jax_scimmf_splitA_immf_q5.npz --no-ref
python scripts/run_szifi_jax_homog_immf.py --out-root $ROOT/szifi_homog/L1_m9_cibshuffle \
  --tag szifi_jax --mmf-type standard --skip-plot \
  --ref $ARCH/L1_m9_cibshuffle/catalogues/szifi_jax_splitA_immf_q5.npz --no-ref
python scripts/run_szifi_jax_homog_immf.py --out-root $ROOT/szifi_homog/L1_m9_cibshuffle \
  --tag szifi_jax_scimmf --mmf-type spectrally_constrained --skip-plot \
  --ref $ARCH/L1_m9_cibshuffle/catalogues/szifi_jax_scimmf_splitA_immf_q5.npz --no-ref
```

sciMMF = `--mmf-type spectrally_constrained` with `deproject_cib=["cib"]`.
Catalogue files keep the archived name `{tag}_splitA_immf_q5.npz`.

**Done 2026-09-05** (logs `logs/szifi_jax_*_regen.log`): L1_m9 iMMF *N*=2602, sciMMF 2867;
L1_m9_cibshuffle iMMF 3119, sciMMF 2979. Per-tile σ: 768 tiles × 25 θ points under
`catalogues/sigma_per_tile_{immf,scimmf}_splitA/`.

Feedback prescriptions (`fgas-8sigma`, `Mstar-1sigma`, `LS8`) use the same
prepare + `--prescription` driver when needed for HILC masks. Not re-run in this
round.

CNC figure (q≥5):

```bash
python scripts/plot_szifi_homog_binned_Nq.py \
  --cats $ROOT/szifi_homog/L1_m9/catalogues/szifi_jax_splitA_immf_q5.npz \
         $ROOT/szifi_homog/L1_m9_cibshuffle/catalogues/szifi_jax_splitA_immf_q5.npz \
         $ROOT/szifi_homog/L1_m9/catalogues/szifi_jax_scimmf_splitA_immf_q5.npz \
         $ROOT/szifi_homog/L1_m9_cibshuffle/catalogues/szifi_jax_scimmf_splitA_immf_q5.npz \
  --labels "iMMF correlated" "iMMF shuffled CIB" "sciMMF correlated" "sciMMF shuffled CIB" \
  --stem szifi_homog_cnc_binned_Nq_qgt5_immf_scimmf_l1m9_cibshuffle
bash scripts/organize_figures.sh
```

## Step 4 — q5 cluster masks for masked HILC

After catalogues exist:

```bash
for rx in L1_m9 fgas-8sigma Mstar-1sigma LS8; do
  python scripts/build_szifi_q5_cluster_mask.py --prescription "$rx"
done
```

Re-run masked HILC configs if masks changed:

```bash
python scripts/run_hilc_prescriptions.py   # skips existing y-maps; delete stale masked dirs first if needed
```

## Step 5 — publication figures

Use `scripts/pub_style.py` helpers (`apply_pub_style`, `savefig` → PDF+PNG @ 300 dpi).

```bash
# HILC prescription suite (under figures/hilc/)
python scripts/plot_hilc_homog_prescriptions.py
python scripts/plot_hilc_combined_deproj_compare.py

# L1 feedback / ILC errors
python scripts/plot_l1_m9_feedback_ratio_ilc_errors.py

# Yang26 cross-check
python scripts/check_l1_m9_yang26_cross.py

# SZiFi CNC / footprint
python scripts/plot_szifi_homog_binned_Nq.py
python scripts/regenerate_ilc_szifi_paper_figs.py
python scripts/regenerate_paper_figures.py

# Tidy topic folders after loose saves at figures/ root
bash scripts/organize_figures.sh
```

## Optional — rebuild L1_m9_cibshuffle totals

Only 100/143 GHz totals need refresh if CIB changed (already done 2026-09-05):

```bash
python scripts/build_l1_m9_cibshuffle_totals.py
```
