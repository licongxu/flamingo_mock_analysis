# FLAMINGO mock CMB paper

LaTeX source and compiled PDF for the software/methods paper documenting
multi-frequency CMB sky simulation from FLAMINGO lightcones.

## Build

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
cd paper
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # resolve refs
```

Or: `latexmk -pdf main.tex`

## Regenerate publication figures

```bash
python scripts/regenerate_paper_figures.py
```

Uses `scripts/pub_style.py` (`text.usetex=True`, readable fonts, no gridlines).
