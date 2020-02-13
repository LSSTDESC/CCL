import numpy as np
import pytest

import pyccl as ccl

COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear',
    mass_function='shethtormen')


@pytest.mark.parametrize('k', [
    1,
    1.0,
    [0.3, 0.5, 10],
    np.array([0.3, 0.5, 10])])
@pytest.mark.parametrize('kind', ['one', 'two', 'total'])
def test_halomodel_power(k, kind):
    a = 0.8

    if kind == 'one':
        func = ccl.onehalo_matter_power
    elif kind == 'two':
        func = ccl.twohalo_matter_power
    else:
        func = ccl.halomodel_matter_power

    pk = func(COSMO, k, a)
    assert np.all(np.isfinite(pk))
    assert np.shape(k) == np.shape(pk)


@pytest.mark.parametrize('m', [
    1e14,
    int(1e14),
    [1e14, 1e15],
    np.array([1e14, 1e15])])
def test_halo_concentration(m):
    a = 0.8
    c = ccl.halo_concentration(COSMO, m, a)
    assert np.all(np.isfinite(c))
    assert np.shape(c) == np.shape(m)


@pytest.mark.parametrize('mf_c', [['shethtormen', 'bhattacharya2011'],
                                  ['shethtormen', 'duffy2008'],
                                  ['shethtormen', 'constant_concentration'],
                                  ['tinker10', 'constant_concentration']])
def test_halomodel_choices_smoke(mf_c):
    mf, c = mf_c
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function='bbks', matter_power_spectrum='linear',
        mass_function=mf, halo_concentration=c)
    a = 0.8
    k = np.geomspace(1E-2, 1, 10)
    p = ccl.twohalo_matter_power(cosmo, k, a)

    assert np.all(np.isfinite(p))


def test_halomodel_choices_raises():
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function='bbks', matter_power_spectrum='linear',
        mass_function='tinker')
    a = 0.8
    k = np.geomspace(1E-2, 1, 10)

    with pytest.raises(ValueError):
        ccl.twohalo_matter_power(cosmo, k, a)


def test_halomodel_power_consistent():
    a = 0.8
    k = np.logspace(-1, 1, 10)
    tot = ccl.halomodel_matter_power(COSMO, k, a)
    one = ccl.onehalo_matter_power(COSMO, k, a)
    two = ccl.twohalo_matter_power(COSMO, k, a)

    assert np.allclose(one + two, tot)
