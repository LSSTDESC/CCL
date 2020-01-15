import numpy as np
import pytest

import pyccl as ccl

COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')

COSMO_NU = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear', m_nu=0.1,
    Omega_k=0.1)

AVALS = [
    1,
    1.0,
    0.8,
    [0.2, 0.4],
    np.array([0.2, 0.4])]


@pytest.mark.parametrize('a', AVALS)
@pytest.mark.parametrize('func', [
    ccl.growth_factor,
    ccl.growth_rate,
    ccl.growth_factor_unnorm,
    ccl.scale_factor_of_chi,
    ccl.comoving_angular_distance,
    ccl.comoving_radial_distance,
    ccl.angular_diameter_distance,
    ccl.luminosity_distance,
    ccl.h_over_h0,
    ccl.distance_modulus,
    ccl.mu_MG,
    ccl.Sig_MG])
def test_background_a_interface(a, func):
    if func is ccl.distance_modulus and np.any(a == 1):
        with pytest.raises(ccl.CCLError):
            func(COSMO, a)
    else:
        val = func(COSMO, a)
        assert np.all(np.isfinite(val))
        assert np.shape(val) == np.shape(a)
        if(func is ccl.angular_diameter_distance):
            val = func(COSMO, a, a)
            assert np.all(np.isfinite(val))
            assert np.shape(val) == np.shape(a)
            if(isinstance(a, float) or isinstance(a, int)):
                val1 = ccl.angular_diameter_distance(COSMO, 1., a)
                val2 = ccl.comoving_angular_distance(COSMO, a)*a
            else:
                val1 = ccl.angular_diameter_distance(COSMO, np.ones(len(a)), a)
                val2 = ccl.comoving_angular_distance(COSMO, a)*a
            assert np.allclose(val1, val2)


@pytest.mark.parametrize('a', AVALS)
@pytest.mark.parametrize('kind', [
    'matter',
    'dark_energy',
    'radiation',
    'curvature',
    'neutrinos_rel',
    'neutrinos_massive'])
def test_background_omega_x(a, kind):
    val = ccl.omega_x(COSMO_NU, a, kind)
    assert np.all(np.isfinite(val))
    assert np.shape(val) == np.shape(a)

    if np.all(a == 1):
        if kind == 'matter':
            val_z0 = (
                COSMO_NU['Omega_b'] +
                COSMO_NU['Omega_c'] +
                COSMO_NU['Omega_nu_mass'])
        elif kind == 'dark_energy':
            val_z0 = COSMO_NU['Omega_l']
        elif kind == 'radiation':
            val_z0 = COSMO_NU['Omega_g']
        elif kind == 'curvature':
            val_z0 = COSMO_NU['Omega_k']
        elif kind == 'neutrinos_rel':
            val_z0 = COSMO_NU['Omega_nu_rel']
        elif kind == 'neutrinos_massive':
            val_z0 = COSMO_NU['Omega_nu_mass']

        assert np.allclose(val, val_z0)


def test_background_omega_x_raises():
    with pytest.raises(ValueError):
        ccl.omega_x(COSMO, 1, 'blah')


@pytest.mark.parametrize('a', AVALS)
@pytest.mark.parametrize('kind', [
    'matter',
    'dark_energy',
    'radiation',
    'curvature',
    'neutrinos_rel',
    'neutrinos_massive'])
@pytest.mark.parametrize('is_comoving', [True, False])
def test_background_rho_x(a, kind, is_comoving):
    val = ccl.rho_x(COSMO_NU, a, kind, is_comoving)
    assert np.all(np.isfinite(val))
    assert np.shape(val) == np.shape(a)


def test_background_rho_x_raises():
    with pytest.raises(ValueError):
        ccl.rho_x(COSMO, 1, 'blah', False)
