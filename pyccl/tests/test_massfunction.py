import numpy as np
import pyccl as ccl
import pytest


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
    mf = ccl.massfunc(COSMO, m, a)
    assert np.all(np.isfinite(mf))
    assert np.shape(mf) == np.shape(m)


@pytest.mark.parametrize('m', [
    1e14,
    int(1e14),
    [1e14, 1e15],
    np.array([1e14, 1e15])])
def test_massfunc_m2r_smoke(m):
    r = ccl.massfunc_m2r(COSMO, m)
    assert np.all(np.isfinite(r))
    assert np.shape(r) == np.shape(m)


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


@pytest.mark.parametrize('m', [
    1e14,
    int(1e14),
    [1e14, 1e15],
    np.array([1e14, 1e15])])
def test_halo_bias_smoke(m):
    a = 0.8
    b = ccl.halo_bias(COSMO, m, a)
    assert np.all(np.isfinite(b))
    assert np.shape(b) == np.shape(m)
