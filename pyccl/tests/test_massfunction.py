import numpy as np
import pytest
import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='halofit')


@pytest.mark.parametrize('m', [
    1e14,
    int(1e14),
    [1e14, 1e15],
    np.array([1e14, 1e15])])
def test_massfunc_smoke(m):
    a = 0.8
    mf = ccl.halos.MassFuncTinker10()(COSMO, m, a)
    assert np.all(np.isfinite(mf))
    assert np.shape(mf) == np.shape(m)


@pytest.mark.parametrize('m', [
    1e14,
    int(1e14),
    [1e14, 1e15],
    np.array([1e14, 1e15])])
def test_sigmaM_smoke(m):
    a = 0.8
    s = ccl.sigmaM(COSMO, m, a)
    assert np.all(np.isfinite(s))
    assert np.shape(s) == np.shape(m)


def test_deltac():
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05,
                          h=0.7, n_s=0.96, sigma8=0.8,
                          transfer_function='bbks')
    # Test EdS value
    dca = 3*(12*np.pi)**(2/3)/20
    dcb = ccl.halos.get_delta_c(cosmo, 1.0, kind='EdS')
    assert np.fabs(dcb/dca-1) < 1E-5

    # Test Mead et al. value
    dca = (1.59+0.0314*np.log(0.8))*(1+0.0123*np.log10(0.3))
    dcb = ccl.halos.get_delta_c(cosmo, 1.0, kind='Mead16')
    assert np.fabs(dcb/dca-1) < 1E-5
