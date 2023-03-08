import numpy as np
import pytest
from unittest import mock
import pyccl as ccl
import sys
from numpy.testing import assert_raises
from pyccl.boltzmann import get_isitgr_pk_lin
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
        transfer_function=tf, mu_0=0.1, sigma_0=0.2)

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
        with assert_raises(ModuleNotFoundError):
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
        transfer_function=tf, mu_0=0.1, sigma_0=0.2)
    sigma8 = ccl.sigma8(cosmo)

    # remake same but now give sigma8
    cosmo_s8 = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=sigma8, n_s=0.96,
        transfer_function=tf, mu_0=0.1, sigma_0=0.2)

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
