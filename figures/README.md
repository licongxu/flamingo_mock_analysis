# Figure layout

Top-level topic folders keep unrelated plots out of the repo root.

| Folder | Contents |
|--------|----------|
| `hilc/` | HILC Compton-y diagnostics (see `hilc/README.md`) |
| `l1_m9/` | L1_m9 feedback-ratio and ILC error-band figures |
| `yang26/` | Cross-comparison with Yang et al. (2026) prescriptions |
| `noise/` | Planck NPIPE vs FFP10 noise galleries and spectra |
| `szifi/` | SZiFi cluster-finding footprint / CNC / zoom plots |

Regenerate HILC prescription plots:

```bash
python scripts/plot_hilc_homog_prescriptions.py
```

Re-tidy loose files at `figures/` root:

```bash
bash scripts/organize_figures.sh
```
