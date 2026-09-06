# Figure layout

Publication-quality figures for the homog pipeline regen (post z_eff=1.9) go here.

| Folder | Contents |
|--------|----------|
| `hilc/` | HILC Compton-y diagnostics (see `hilc/README.md`) |
| `l1_m9/` | L1_m9 feedback-ratio and ILC error-band figures |
| `yang26/` | Cross-comparison with Yang et al. (2026) prescriptions |
| `noise/` | Planck NPIPE vs FFP10 noise galleries and spectra |
| `szifi/` | SZiFi cluster-finding footprint / CNC / zoom plots |
| `archive/demo/` | Pre-regeneration demo figures (2026-09-05) |

Style: `scripts/pub_style.py` — LaTeX labels, 300 dpi, PDF + PNG, no gridlines.

Regenerate HILC prescription plots:

```bash
python scripts/plot_hilc_homog_prescriptions.py
```

Full pipeline order: `docs/regeneration_pipeline.md`.

Re-tidy loose files at `figures/` root:

```bash
bash scripts/organize_figures.sh
```
