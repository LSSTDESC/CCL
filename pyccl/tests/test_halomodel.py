import numpy as np
import pytest

import pyccl as ccl

with pytest.warns(ccl.CCLDeprecationWarning):
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

    with pytest.warns(ccl.CCLDeprecationWarning):
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
    # Deprecated.
    with pytest.warns(ccl.CCLDeprecationWarning):
        c = ccl.halo_concentration(COSMO, m, a)
    assert np.all(np.isfinite(c))
    assert np.shape(c) == np.shape(m)


def get_pk_new(mf, c, cosmo, a, k, get_1h, get_2h):
    mdef = ccl.halos.MassDef('vir', 'matter')
    if mf == 'shethtormen':
        hmf = ccl.halos.MassFuncSheth99(
            mass_def=mdef, mass_def_strict=False, use_delta_c_fit=True)
        hbf = ccl.halos.HaloBiasSheth99(mass_def=mdef, mass_def_strict=False)
    elif mf == 'tinker10':
        hmf = ccl.halos.MassFuncTinker10(mass_def=mdef, mass_def_strict=False)
        hbf = ccl.halos.HaloBiasTinker10(mass_def=mdef, mass_def_strict=False)

    if c == 'constant_concentration':
        cc = ccl.halos.ConcentrationConstant(4., mass_def=mdef)
    elif c == 'duffy2008':
        cc = ccl.halos.ConcentrationDuffy08(mass_def=mdef)
    elif c == 'bhattacharya2011':
        cc = ccl.halos.ConcentrationBhattacharya13(mass_def=mdef)
    prf = ccl.halos.HaloProfileNFW(c_m_relation=cc)
    hmc = ccl.halos.HMCalculator(mass_function=hmf, halo_bias=hbf,
                                 mass_def=mdef)
    p = ccl.halos.halomod_power_spectrum(cosmo, hmc, k, a,
                                         prf, normprof=True,
                                         get_1h=get_1h,
                                         get_2h=get_2h)
    return p


@pytest.mark.parametrize('mf_c', [['shethtormen', 'bhattacharya2011'],
                                  ['shethtormen', 'duffy2008'],
                                  ['shethtormen', 'constant_concentration'],
                                  ['tinker10', 'constant_concentration']])
def test_halomodel_choices_smoke(mf_c):
    mf, c = mf_c
    with pytest.warns(ccl.CCLDeprecationWarning):
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
    with pytest.warns(ccl.CCLDeprecationWarning):
        p = ccl.twohalo_matter_power(cosmo, k, a)
    pb = get_pk_new(mf, c, cosmo, a, k, False, True)

    assert np.all(np.isfinite(p))
    assert np.allclose(p, pb)


def test_halomodel_choices_raises():
    with pytest.warns(ccl.CCLDeprecationWarning):
        cosmo = ccl.Cosmology(
            Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
            transfer_function='bbks', matter_power_spectrum='linear',
            mass_function='tinker')
    a = 0.8
    k = np.geomspace(1E-2, 1, 10)

    with pytest.raises(ValueError):
        with pytest.warns(ccl.CCLDeprecationWarning):
            ccl.twohalo_matter_power(cosmo, k, a)


def test_halomodel_power_consistent():
    a = 0.8
    k = np.logspace(-1, 1, 10)
    # These are all deprecated.
    with pytest.warns(ccl.CCLDeprecationWarning):
        tot = ccl.halomodel_matter_power(COSMO, k, a)
    with pytest.warns(ccl.CCLDeprecationWarning):
        one = ccl.onehalo_matter_power(COSMO, k, a)
    with pytest.warns(ccl.CCLDeprecationWarning):
        two = ccl.twohalo_matter_power(COSMO, k, a)

    assert np.allclose(one + two, tot)
