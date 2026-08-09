"""Needlet / harmonic ILC Compton-y reconstruction on FLAMINGO mock skies.

McCarthy & Hill (2024) style ILC (arXiv:2307.01043) with the locally
installed `pyilc <https://github.com/licongxu/pyilc/tree/agent_evolve>`_
package (``agent_evolve`` branch), whose constrained-ILC weight solves run on
multi-backend linear algebra (numpy / numba / JAX / CuPy). On GPU hosts the
JAX backend is the default.

Pipeline stages (see also the ``flamingo-ilc`` command):

1. :mod:`flamingo_mock.ilc.prepare` — coadd components, apply channel beams,
   add Planck NPIPE noise splits A/B.
2. :mod:`flamingo_mock.ilc.config` — write pyILC YAML configs (HILC / NILC).
3. :mod:`flamingo_mock.ilc.run` — execute pyILC from a YAML config.
4. :mod:`flamingo_mock.ilc.validate` — beam-deconvolved validation vs truth y.

Heavy dependencies (healpy, pyilc, matplotlib) are imported inside functions
so that ``import flamingo_mock.ilc`` stays cheap.
"""
