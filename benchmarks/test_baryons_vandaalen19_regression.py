"""Regression test for the van Daalen+2019 baryonic model.

The reference data in ``baryons_vd19_fk_500c.txt`` were generated once using
``benchmarks/data/codes/make_baryons_vd19_benchmark.py`` with the same
cosmology and BaryonsvanDaalen19 configuration. This is a frozen regression
target (not an external benchmark) intended to catch unintended changes in
the implementation or numerical settings.
"""

from pathlib import Path

import numpy as np

import pyccl as ccl


VD19_REGRESSION_TOLERANCE = 1e-8

DATA_PATH = Path(__file__).parent / "data" / "baryons_vd19_fk_500c.txt"
DATA = np.loadtxt(DATA_PATH)


def _make_cosmo() -> ccl.Cosmology:
    """Cosmology used to generate the vd19 regression reference file."""
    return ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        A_s=2.2e-9,
        n_s=0.96,
        Neff=3.046,
        mass_split="normal",
        m_nu=0.0,
        Omega_g=0.0,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
    )


def _check_vd19_baryons(cosmo: ccl.Cosmology, baryons: ccl.BaryonsvanDaalen19) -> None:
    """Checks that the vd19 boost factor matches the frozen regression data."""
    # First column is k in h/Mpc, second is boost factor f(k)
    k_hmpc = DATA[:, 0]
    fk_ref = DATA[:, 1]
    a = 1.0

    # Convert to 1/Mpc for the model: k = k_hmpc * h
    k = k_hmpc * cosmo["h"]

    fk_ccl = baryons.boost_factor(cosmo, k, a)
    err = np.abs(fk_ref / fk_ccl - 1.0)
    np.testing.assert_allclose(
        err, 0.0, atol=VD19_REGRESSION_TOLERANCE, rtol=0.0
    )


def test_baryons_vd19_500c():
    """Tests that the vd19 baryonic model is stable for the 500c config."""
    cosmo = _make_cosmo()
    baryons = ccl.BaryonsvanDaalen19(fbar=0.7, mass_def="500c")
    _check_vd19_baryons(cosmo, baryons)
