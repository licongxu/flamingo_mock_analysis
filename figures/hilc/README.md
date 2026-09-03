# HILC figure layout

## Per prescription (`L1_m9`, `fgas-8sigma`, `Mstar-1sigma`, `LS8`)

Each subdirectory holds **no-deprojection** HILC diagnostics from
`scripts/plot_hilc_homog_prescriptions.py`:

| File | Content |
|------|---------|
| `r1xr2_fig9_fullsky.png` | r1×r2 split cross + CIB/CMB/noise residuals (full sky) |
| `r1xr2_fig9_q5masked.png` | same, with **that prescription's** q>5 iMMF cluster holes |
| `y_vs_truth_fullsky.png` | HILC y vs FLAMINGO truth y |
| `y_vs_truth_q5masked.png` | masked variant |

Masked runs use `szifi_homog/<prescription>/catalogues/fullsky_splitA_immf_q5.npz`
(N clusters differs per case).

## Combined overlays (repo root of this folder)

- `r1xr2_fig9_fullsky_all.png` / `r1xr2_fig9_q5masked_all.png` — four-panel stacks
- `r1xr2_compare_fullsky.png` / `r1xr2_compare_q5masked.png` — overlay curves

## L1_m9 deprojection suite (`l1_m9_deproj_suite/`)

Earlier L1_m9-only plots (CIB deproj, auto Fig. 9, weight comparisons) from
`scripts/plot_hilc_homog_*.py`. Regenerate with `scripts/run_hilc_execute_replot_all.sh`.
