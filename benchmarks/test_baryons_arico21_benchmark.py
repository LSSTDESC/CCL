"""Benchmark tests for the Arico21 BACCOemu baryonic correction model."""

from pathlib import Path

import numpy as np

import pyccl as ccl

DATA_PATH = Path(__file__).parent / "data" / "baccoemu_baryons_fk.txt"
DATA = np.loadtxt(DATA_PATH)

BEMBAR_TOLERANCE = 1e-3


def _check_baccoemu_baryons(
        cosmo: ccl.Cosmology,
        baryons: ccl.BaryonsBaccoemu
) -> None:
    """Checks that the BACCOemu boost factor matches the reference fk(k)."""
    k = DATA[:, 0] * cosmo["h"]
    fk_ref = DATA[:, 1]
    a = 1.0

    fk_ccl = baryons.boost_factor(cosmo, k, a)
    err = np.abs(fk_ref / fk_ccl - 1.0)
    np.testing.assert_allclose(err, 0.0, atol=BEMBAR_TOLERANCE, rtol=0.0)


def test_baccoemu_baryons_sigma8():
    """Tests that BACCOemu baryons match the reference for sigma8 cosmology."""
    baryons = ccl.BaryonsBaccoemu()
    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.05,
        h=0.67,
        sigma8=0.83,
        n_s=0.96,
        Neff=3.046,
        mass_split="normal",
        m_nu=0.1,
        Omega_g=0.0,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
    )

    _check_baccoemu_baryons(cosmo, baryons)


def test_baccoemu_baryons_a_s():
    """Tests that BACCOemu baryons match reference fk(k) for As cosmology."""
    baryons = ccl.BaryonsBaccoemu()
    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.05,
        h=0.67,
        A_s=2.2194e-9,
        n_s=0.96,
        Neff=3.046,
        mass_split="normal",
        m_nu=0.1,
        Omega_g=0.0,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
    )

    _check_baccoemu_baryons(cosmo, baryons)
