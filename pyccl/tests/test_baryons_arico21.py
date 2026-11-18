"""Unit tests for the BACCO-based baryonic correction model in CCL."""

from __future__ import annotations

import pytest

import numpy as np

import pyccl as ccl


BEMULIN_TOLERANCE = 1e-3
BEMUNL_TOLERANCE = 5e-3
BEMBAR_TOLERANCE = 1e-3


def _bacco_cosmo(**extra_kwargs):
    """Returns a BACCO-compatible cosmology, allowing overrides."""
    base = dict(
        Omega_c=0.27,
        Omega_b=0.05,
        h=0.67,
        sigma8=0.83,
        n_s=0.96,
        Neff=3.046,
        mass_split="normal",
        m_nu=0.1,
        Omega_g=0,
        Omega_k=0,
        w0=-1,
        wa=0,
    )
    base.update(extra_kwargs)
    return ccl.Cosmology(**base)


@pytest.mark.parametrize(
    "k_in",
    [
        np.array([0.01]),
        np.array([0.01, 0.1, 1.0]),
        np.logspace(-2, 0.5, 5),
    ],
)
def test_bacco_baryons_boost_smoke(k_in):
    """Tests that boost_factor runs and returns finite values."""
    baryons = ccl.BaryonsBaccoemu()
    cosmo = _bacco_cosmo()
    a = 1.0

    fk = baryons.boost_factor(cosmo, k_in, a)
    assert np.all(np.isfinite(fk))
    assert np.shape(fk) == np.shape(k_in)


def test_bacco_baryons_boost_vs_include():
    """Tests that boost_factor and include_baryonic_effects are consistent."""
    baryons = ccl.BaryonsBaccoemu()
    nlpkemu = ccl.BaccoemuNonlinear()

    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.05,
        h=0.67,
        sigma8=0.83,
        n_s=0.96,
        Neff=3.046,
        mass_split="normal",
        m_nu=0.1,
        Omega_g=0,
        Omega_k=0,
        w0=-1,
        wa=0,
        matter_power_spectrum=nlpkemu,
    )

    k = np.logspace(-2, 0.5, 100)
    a = 1.0

    fk_boost = baryons.boost_factor(cosmo, k, a)

    pk_dmo = cosmo.get_nonlin_power()
    pk_bary = baryons.include_baryonic_effects(cosmo, pk_dmo)
    fk_ratio = pk_bary(k, a) / pk_dmo(k, a)

    err = np.abs(fk_ratio / fk_boost - 1.0)
    assert np.allclose(err, 0.0, atol=BEMBAR_TOLERANCE, rtol=0.0)


def test_bacco_baryons_update_params():
    """Tests that update_parameters correctly updates bcm_params."""
    baryons = ccl.BaryonsBaccoemu()
    baryons.update_parameters(log10_M_c=12.7, log10_eta=-0.4)
    assert baryons.bcm_params["M_c"] == 12.7
    assert baryons.bcm_params["eta"] == -0.4


def test_bacco_baryons_a_range():
    """Tests that boost_factor raises outside valid a-range."""
    baryons = ccl.BaryonsBaccoemu()
    cosmo = ccl.CosmologyVanillaLCDM()
    k = 1e-1

    with pytest.raises(ValueError):
        baryons.boost_factor(cosmo, k, baryons.a_min * 0.9)


def test_bacco_baryons_as_sigma8():
    """Tests that boost_factor is consistent for A_s and sigma8 cosmologies."""
    baryons = ccl.BaryonsBaccoemu()

    cosmo1 = _bacco_cosmo()
    cosmo2 = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.05,
        h=0.67,
        A_s=2.2194e-09,
        n_s=0.96,
        Neff=3.046,
        mass_split="normal",
        m_nu=0.1,
        Omega_g=0,
        Omega_k=0,
        w0=-1,
        wa=0,
    )

    k = np.logspace(-2, 0.5, 100)
    a = 1.0

    fk1 = baryons.boost_factor(cosmo1, k, a)
    fk2 = baryons.boost_factor(cosmo2, k, a)

    err = np.abs(fk1 / fk2 - 1.0)
    assert np.allclose(err, 0.0, atol=BEMUNL_TOLERANCE, rtol=0.0)


def test_bacco_baryons_include_returns_pk2d():
    """Tests that include_baryonic_effects returns a Pk2D object."""
    baryons = ccl.BaryonsBaccoemu()
    cosmo = _bacco_cosmo()

    # Restrict a-grid to the emulator's valid range [a_min, 1]
    a_min = float(baryons.a_min)
    a_arr = np.linspace(a_min, 1.0, 5)
    lk_arr = np.linspace(np.log(1e-3), np.log(1.0), 10)

    # Simple baseline P(k, a) = 1 everywhere on this grid
    pk_arr = np.ones((a_arr.size, lk_arr.size))

    pk_dmo = ccl.Pk2D(
        a_arr=a_arr,
        lk_arr=lk_arr,
        pk_arr=pk_arr,
        is_logp=False,
        extrap_order_lok=1,
        extrap_order_hik=1,
    )

    pk_bary = baryons.include_baryonic_effects(cosmo, pk_dmo)
    assert isinstance(pk_bary, ccl.Pk2D)


def test_bacco_baryons_in_cosmology_raises_outside_a_range():
    """Tests that get_nonlin_power raises if baryons are included."""
    baryons = ccl.BaryonsBaccoemu()
    cosmo_wb = _bacco_cosmo(baryonic_effects=baryons)

    with pytest.raises(ValueError):
        cosmo_wb.get_nonlin_power()
