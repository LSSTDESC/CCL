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
    from pyccl.pyutils import assert_warns
    a = 0.8

    if kind == 'one':
        func = ccl.onehalo_matter_power
    elif kind == 'two':
        func = ccl.twohalo_matter_power
    else:
        func = ccl.halomodel_matter_power

    pk = assert_warns(ccl.CCLWarning, func, COSMO, k, a)
    assert np.all(np.isfinite(pk))
    assert np.shape(k) == np.shape(pk)


@pytest.mark.parametrize('m', [
    1e14,
    int(1e14),
    [1e14, 1e15],
    np.array([1e14, 1e15])])
def test_halo_concentration(m):
    from pyccl.pyutils import assert_warns
    a = 0.8
    # Deprecated.
    c = assert_warns(
        ccl.CCLWarning,
        ccl.halo_concentration, COSMO, m, a)
    assert np.all(np.isfinite(c))
    assert np.shape(c) == np.shape(m)


def get_pk_new(mf, c, cosmo, a, k, get_1h, get_2h):
    mdef = ccl.halos.MassDef('vir', 'matter')
    if mf == 'shethtormen':
        hmf = ccl.halos.MassFuncSheth99(cosmo, mdef,
                                        mass_def_strict=False,
                                        use_delta_c_fit=True)
        hbf = ccl.halos.HaloBiasSheth99(cosmo, mass_def=mdef,
                                        mass_def_strict=False)
    elif mf == 'tinker10':
        hmf = ccl.halos.MassFuncTinker10(cosmo, mdef,
                                         mass_def_strict=False)
        hbf = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mdef,
                                         mass_def_strict=False)

    if c == 'constant_concentration':
        cc = ccl.halos.ConcentrationConstant(4., mdef)
    elif c == 'duffy2008':
        cc = ccl.halos.ConcentrationDuffy08(mdef)
    elif c == 'bhattacharya2011':
        cc = ccl.halos.ConcentrationBhattacharya13(mdef)
    prf = ccl.halos.HaloProfileNFW(cc)
    hmc = ccl.halos.HMCalculator(cosmo, hmf, hbf, mdef)
    p = ccl.halos.halomod_power_spectrum(cosmo, hmc, k, a,
                                         prf, normprof1=True,
                                         get_1h=get_1h,
                                         get_2h=get_2h)
    return p


@pytest.mark.parametrize('mf_c', [['shethtormen', 'bhattacharya2011'],
                                  ['shethtormen', 'duffy2008'],
                                  ['shethtormen', 'constant_concentration'],
                                  ['tinker10', 'constant_concentration']])
def test_halomodel_choices_smoke(mf_c):
    from pyccl.pyutils import assert_warns
    mf, c = mf_c
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function='bbks', matter_power_spectrum='linear',
        mass_function=mf, halo_concentration=c)
    a = 0.8
    k = np.geomspace(1E-2, 1, 10)
    # Deprecated
    # TODO: Convert this and other places to using the non-deprecated syntax
    # Or, since this wasn't already done, maybe this is a useful convenience
    # function?
    p = assert_warns(ccl.CCLWarning, ccl.twohalo_matter_power, cosmo, k, a)
    pb = get_pk_new(mf, c, cosmo, a, k, False, True)

    assert np.all(np.isfinite(p))
    assert np.allclose(p, pb)


def test_halomodel_choices_raises():
    from pyccl.pyutils import assert_warns
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function='bbks', matter_power_spectrum='linear',
        mass_function='tinker')
    a = 0.8
    k = np.geomspace(1E-2, 1, 10)

    with pytest.raises(ValueError):
        assert_warns(ccl.CCLWarning, ccl.twohalo_matter_power, cosmo, k, a)


def test_halomodel_power_consistent():
    from pyccl.pyutils import assert_warns
    a = 0.8
    k = np.logspace(-1, 1, 10)
    # These are all deprecated.
    tot = assert_warns(
        ccl.CCLWarning,
        ccl.halomodel_matter_power, COSMO, k, a)
    one = assert_warns(
        ccl.CCLWarning,
        ccl.onehalo_matter_power, COSMO, k, a)
    two = assert_warns(
        ccl.CCLWarning,
        ccl.twohalo_matter_power, COSMO, k, a)

    assert np.allclose(one + two, tot)
