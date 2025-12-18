"""Benchmark tests for the Schneider2015 baryonic correction model."""

from pathlib import Path

import numpy as np

import pyccl as ccl


BCM_TOLERANCE = 1e-4

DATA_WITH_BAR_PATH = Path(__file__).parent / "data" / "w_baryonspk_nl.dat"
DATA_NO_BAR_PATH = Path(__file__).parent / "data" / "wo_baryonspk_nl.dat"

DATA_WITH_BAR = np.loadtxt(DATA_WITH_BAR_PATH)
DATA_NO_BAR = np.loadtxt(DATA_NO_BAR_PATH)


def test_bcm():
    """Tests that the Schneider15 model matches the reference fk(k)."""
    cosmo = ccl.Cosmology(
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

    k = DATA_WITH_BAR[:, 0] * cosmo["h"]
    a = 1.0

    bar = ccl.BaryonsSchneider15(log10Mc=14.0)

    # 1) Direct boost-factor comparison
    fbcm = bar.boost_factor(cosmo, k, a)
    ratio_ref = DATA_WITH_BAR[:, 1] / DATA_NO_BAR[:, 1]
    err = np.abs(ratio_ref / fbcm - 1.0)
    np.testing.assert_allclose(err, 0.0, atol=BCM_TOLERANCE, rtol=0.0)

    # 2) Full P(k) with and without baryons
    cosmo.compute_nonlin_power()
    pk_nobar = cosmo.get_nonlin_power()
    pk_wbar = bar.include_baryonic_effects(cosmo, pk_nobar)

    ratio = pk_wbar(k, a) / pk_nobar(k, a)
    err = np.abs(ratio_ref / ratio - 1.0)
    np.testing.assert_allclose(err, 0.0, atol=BCM_TOLERANCE, rtol=0.0)
