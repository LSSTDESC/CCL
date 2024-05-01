import numpy as np
import pyccl as ccl

import pytest

BEMULIN_TOLERANCE = 1e-3
BEMUNL_TOLERANCE = 5e-3
BEMBAR_TOLERANCE = 1e-3


def test_baccoemu_linear_As_sigma8():
    bemu = ccl.BaccoemuLinear()
    cosmo1 = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.05,
        h=0.67,
        sigma8=0.83,
        n_s=0.96,
        Neff=3.046,
        mass_split='normal',
        m_nu=0.1,
        Omega_g=0,
        Omega_k=0,
        w0=-1,
        wa=0)

    cosmo2 = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.05,
        h=0.67,
        A_s=2.2194e-09,
        n_s=0.96,
        Neff=3.046,
        mass_split='normal',
        m_nu=0.1,
        Omega_g=0,
        Omega_k=0,
        w0=-1,
        wa=0)

    k1, pk1 = bemu.get_pk_at_a(cosmo1, 1.0)
    k2, pk2 = bemu.get_pk_at_a(cosmo2, 1.0)

    err = np.abs(pk1 / pk2 - 1)
    assert np.allclose(err, 0, atol=BEMULIN_TOLERANCE, rtol=0)


def test_baccoemu_nonlinear_As_sigma8():
    bemu = ccl.BaccoemuNonlinear()
    cosmo1 = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.05,
        h=0.67,
        sigma8=0.83,
        n_s=0.96,
        Neff=3.046,
        mass_split='normal',
        m_nu=0.1,
        Omega_g=0,
        Omega_k=0,
        w0=-1,
        wa=0)

    cosmo2 = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.05,
        h=0.67,
        A_s=2.2194e-09,
        n_s=0.96,
        Neff=3.046,
        mass_split='normal',
        m_nu=0.1,
        Omega_g=0,
        Omega_k=0,
        w0=-1,
        wa=0)

    k1, pk1 = bemu.get_pk_at_a(cosmo1, 1.0)
    k2, pk2 = bemu.get_pk_at_a(cosmo2, 1.0)

    err = np.abs(pk1 / pk2 - 1)
    assert np.allclose(err, 0, atol=BEMUNL_TOLERANCE, rtol=0)


def test_baccoemu_baryons_boost():
    baryons = ccl.BaryonsBaccoemu()
    nlpkemu = ccl.BaccoemuNonlinear()
    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.05,
        h=0.67,
        sigma8=0.83,
        n_s=0.96,
        Neff=3.046,
        mass_split='normal',
        m_nu=0.1,
        Omega_g=0,
        Omega_k=0,
        w0=-1,
        wa=0,
        matter_power_spectrum=nlpkemu)

    k = np.logspace(-2, 0.5, 100)
    cclfk = baryons.boost_factor(cosmo, k, 1)
    pk_gro = cosmo.get_nonlin_power()
    pk_bcm = baryons.include_baryonic_effects(cosmo, pk_gro)
    fk = pk_bcm(k, 1) / pk_gro(k, 1)
    err = np.abs(fk / cclfk - 1)
    assert np.allclose(err, 0, atol=BEMBAR_TOLERANCE, rtol=0)


def test_baccoemu_baryons_changepars():
    baryons = ccl.BaryonsBaccoemu()
    baryons.update_parameters(log10_M_c=12.7, log10_eta=-0.4)
    assert ((baryons.bcm_params['M_c'] == 12.7)
            & (baryons.bcm_params['eta'] == -0.4))


def test_baccoemu_baryons_a_range():
    baryons = ccl.BaryonsBaccoemu()
    cosmo = ccl.CosmologyVanillaLCDM()
    k = 1e-1
    with pytest.raises(ValueError):
        baryons.boost_factor(cosmo, k, baryons.a_min * 0.9)


def test_baccoemu_baryons_As_sigma8():
    baryons = ccl.BaryonsBaccoemu()
    cosmo1 = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.05,
        h=0.67,
        sigma8=0.83,
        n_s=0.96,
        Neff=3.046,
        mass_split='normal',
        m_nu=0.1,
        Omega_g=0,
        Omega_k=0,
        w0=-1,
        wa=0)

    cosmo2 = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.05,
        h=0.67,
        A_s=2.2194e-09,
        n_s=0.96,
        Neff=3.046,
        mass_split='normal',
        m_nu=0.1,
        Omega_g=0,
        Omega_k=0,
        w0=-1,
        wa=0)

    k = np.logspace(-2, 0.5, 100)
    fk1 = baryons.boost_factor(cosmo1, k, 1)
    fk2 = baryons.boost_factor(cosmo2, k, 1)

    err = np.abs(fk1 / fk2 - 1)
    assert np.allclose(err, 0, atol=BEMUNL_TOLERANCE, rtol=0)
