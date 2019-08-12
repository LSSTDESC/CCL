import numpy as np
import pytest

import pyccl as ccl
from pyccl import CCLError


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='halofit')


@pytest.mark.parametrize('k', [
    1,
    1.0,
    [0.3, 0.5, 10],
    np.array([0.3, 0.5, 10])])
def test_linear_power_smoke(k):
    a = 0.8
    pk = ccl.linear_matter_power(COSMO, k, a)
    assert np.all(np.isfinite(pk))
    assert np.shape(pk) == np.shape(k)


@pytest.mark.parametrize('k', [
    1,
    1.0,
    [0.3, 0.5, 10],
    np.array([0.3, 0.5, 10])])
def test_nonlin_power_smoke(k):
    a = 0.8
    pk = ccl.nonlin_matter_power(COSMO, k, a)
    assert np.all(np.isfinite(pk))
    assert np.shape(pk) == np.shape(k)


@pytest.mark.parametrize('r', [
    1,
    1.0,
    [0.3, 0.5, 10],
    np.array([0.3, 0.5, 10])])
def test_sigmaR_smoke(r):
    a = 0.8
    sig = ccl.sigmaR(COSMO, r, a)
    assert np.all(np.isfinite(sig))
    assert np.shape(sig) == np.shape(r)


@pytest.mark.parametrize('r', [
    1,
    1.0,
    [0.3, 0.5, 10],
    np.array([0.3, 0.5, 10])])
def test_sigmaV_smoke(r):
    a = 0.8
    sig = ccl.sigmaV(COSMO, r, a)
    assert np.all(np.isfinite(sig))
    assert np.shape(sig) == np.shape(r)


def test_sigma8_consistent():
    assert np.allclose(ccl.sigma8(COSMO), COSMO['sigma8'])
    assert np.allclose(ccl.sigmaR(COSMO, 8 / COSMO['h'], 1), COSMO['sigma8'])


@pytest.mark.parametrize('tf,pk,m_nu', [
    ('bbks', 'linear', 0.06),
    ('eisenstein_hu', 'linear', 0.06),
    # ('boltzmann_class', 'emu', 0.06), - this case is slow and not needed
    (None, 'emu', 0.06),
    ('bbks', 'emu', 0.06),
    ('eisenstein_hu', 'emu', 0.06),
])
def test_transfer_matter_power_nu_raises(tf, pk, m_nu):
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function=tf, matter_power_spectrum=pk, m_nu=m_nu)

    with pytest.raises(CCLError):
        ccl.linear_matter_power(cosmo, 1, 1)

    with pytest.raises(CCLError):
        ccl.nonlin_matter_power(cosmo, 1, 1)


@pytest.mark.parametrize('tf,pk', [
    ('bbks', 'linear'),
    ('eisenstein_hu', 'linear'),
    ('bbks', 'halofit'),
    ('eisenstein_hu', 'halofit'),
    (None, 'emu'),
])
def test_transfer_matter_power_mu_sigma_raises(tf, pk):
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function=tf, matter_power_spectrum=pk, mu_0=0.1, sigma_0=0.1)

    with pytest.raises(CCLError):
        ccl.linear_matter_power(cosmo, 1, 1)

    with pytest.raises(CCLError):
        ccl.nonlin_matter_power(cosmo, 1, 1)
