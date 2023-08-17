import numpy as np
import pytest
from unittest import mock
import pyccl as ccl
import sys
from pyccl.boltzmann import get_isitgr_pk_lin
from pyccl.modified_gravity import MuSigmaMG

try:
    from importlib import reload
except ImportError:
    pass  # in 2.7, reload is a global function.


@pytest.mark.parametrize('tf', [
    'bbks', 'eisenstein_hu', 'boltzmann_class', 'boltzmann_camb',
    'boltzmann_isitgr'])
def test_power_mu_sigma_sigma8norm(tf):
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function=tf)

    cosmo_musig = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function=tf,
        mg_parametrization=MuSigmaMG(mu_0=0.1, sigma_0=0.2))

    # make sure sigma8 is correct
    assert np.allclose(ccl.sigma8(cosmo_musig), 0.8)

    if tf != 'boltzmann_isitgr':
        # make sure P(k) ratio is right
        a = 0.8
        gfac = (
            ccl.growth_factor(cosmo, a) / ccl.growth_factor(cosmo_musig, a))**2
        pk_rat = (
            ccl.linear_matter_power(cosmo, 1e-4, a) /
            ccl.linear_matter_power(cosmo_musig, 1e-4, a))
        assert np.allclose(pk_rat, gfac)

    with mock.patch.dict(sys.modules, {'isitgr': None}):
        with pytest.raises(ModuleNotFoundError):
            get_isitgr_pk_lin(cosmo)
    # Importing ccl without isitgr is fine.  No ModuleNotFoundError triggered.
    with mock.patch.dict(sys.modules, {'isitgr': None}):
        reload(ccl.boltzmann)


@pytest.mark.parametrize('tf', [
    'boltzmann_class', 'boltzmann_camb', 'boltzmann_isitgr'])
def test_power_mu_sigma_sigma8norm_norms_consistent(tf):
    # make a cosmo with A_s
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2e-9, n_s=0.96,
        transfer_function=tf,
        mg_parametrization=MuSigmaMG(mu_0=0.1, sigma_0=0.2))
    sigma8 = ccl.sigma8(cosmo)

    # remake same but now give sigma8
    cosmo_s8 = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=sigma8, n_s=0.96,
        transfer_function=tf,
        mg_parametrization=MuSigmaMG(mu_0=0.1, sigma_0=0.2))

    # make sure they come out the same-ish
    assert np.allclose(ccl.sigma8(cosmo), ccl.sigma8(cosmo_s8))

    if tf != 'boltzmann_isitgr':
        # and that the power spectra look right
        a = 0.8
        gfac = (
            ccl.growth_factor(cosmo, a) / ccl.growth_factor(cosmo_s8, a))**2
        pk_rat = (
            ccl.linear_matter_power(cosmo, 1e-4, a) /
            ccl.linear_matter_power(cosmo_s8, 1e-4, a))
        assert np.allclose(pk_rat, gfac)


def test_nonlin_camb_MG_error():
    Omega_c = 0.25
    Omega_b = 0.05
    n_s = 0.97
    h = 0.7

    ccl_cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h, m_nu=0.0,
                              A_s=2.1e-9, n_s=n_s,
                              transfer_function="boltzmann_camb",
                              matter_power_spectrum="camb",
                              mg_parametrization=MuSigmaMG(
                                  mu_0=0.1, sigma_0=0.2))

    k = np.logspace(-3, 1, 10)

    # Check that non-linear power spectrum isn't being used with sigma8
    with pytest.raises(ValueError):
        ccl.nonlin_matter_power(ccl_cosmo, k, 1.0)
