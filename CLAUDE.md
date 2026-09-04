# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## Project-specific instructions

**Branch:** `szifi_branch` — MMF cluster finding on FLAMINGO mock Planck skies with **SZiFi**. Needlet ILC is on `needlet_ilc`; both share on-disk data under `/rds/rds-lxu/flamingo/integrated_maps_synthetic/`.

### Environment

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
pip install -e .   # this repo (flamingo-mock)
```

### SZiFi (GPU)

Use the **GPU-capable fork** on this machine — not upstream PyPI:

```bash
cd /scratch/scratch-lxu/agent_dev/auto_research_agent/szifi   # agent_evolve branch
pip install -e ".[accel]"   # pulls jax; cmbagent_env already has CUDA jax
export SZIFI_ARRAY_BACKEND=jax   # or auto (prefers GPU when present)
```

Host has 2× RTX PRO 6000. Hot paths (MMF convolutions, linear algebra) dispatch through `szifi/backend.py`; results stay NumPy on return. Fallback: `SZIFI_ARRAY_BACKEND=numpy`. Papers: `ref_paper/szifi_refa.pdf` (iMMF), `szifi_refb.pdf` (sciMMF). Planck example: `test_files/run_szifi_planck.py`.

Data dictionaries: `data_description.md`, `noise_description.md`.

### On-disk paths

| Path | Contents |
|------|----------|
| `…/components/{cmb,tsz,ksz,cib}/` | Synthetic signal maps |
| `…/planck_noise/npipe/{freq}GHz/{A,B,full}/` | NPIPE noise MCs, Nside=2048 |
| `…/masks/pr4_nilc/Masks.fits` | PR4 masks: field 1=GAL, 2=PS |
| `…/ilc/gal_ps_mask_nside2048.fits` | Pre-built GAL×PS binary mask |

Do not commit FITS/HDF5 to git.

---

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
