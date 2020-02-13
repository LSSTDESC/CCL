import numpy as np
import pytest

import pyccl as ccl
from pyccl import CCLError, CCLWarning


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='halofit')
COSMO_HM = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='halo_model',
    mass_function='shethtormen')


def test_halomod_f2d_copy():
    mdef = ccl.halos.MassDef('vir', 'matter')
    hmf = ccl.halos.MassFuncSheth99(COSMO_HM, mdef,
                                    mass_def_strict=False,
                                    use_delta_c_fit=True)
    hbf = ccl.halos.HaloBiasSheth99(COSMO_HM, mass_def=mdef,
                                    mass_def_strict=False)
    cc = ccl.halos.ConcentrationDuffy08(mdef)
    prf = ccl.halos.HaloProfileNFW(cc)
    hmc = ccl.halos.HMCalculator(COSMO_HM, hmf, hbf, mdef)
    pk2d = ccl.halos.halomod_Pk2D(COSMO_HM, hmc, prf, normprof1=True)
    psp_new = pk2d.psp
    # This just triggers the internal calculation
    pk_old = ccl.nonlin_matter_power(COSMO_HM, 1., 0.8)
    pk_new = pk2d.eval(1., 0.8, COSMO_HM)
    psp_old = COSMO_HM.cosmo.data.p_nl
    assert psp_new.lkmin == psp_old.lkmin
    assert psp_new.lkmax == psp_old.lkmax
    assert psp_new.amin == psp_old.amin
    assert psp_new.amax == psp_old.amax
    assert psp_new.is_factorizable == psp_old.is_factorizable
    assert psp_new.is_k_constant == psp_old.is_k_constant
    assert psp_new.is_a_constant == psp_old.is_a_constant
    assert psp_new.is_log == psp_old.is_log
    assert psp_new.growth_factor_0 == psp_old.growth_factor_0
    assert psp_new.growth_exponent == psp_old.growth_exponent
    assert psp_new.extrap_order_lok == psp_old.extrap_order_lok
    assert psp_new.extrap_order_hik == psp_old.extrap_order_hik
    assert pk_old == pk_new


@pytest.mark.parametrize('k', [
    1,
    1.0,
    [0.3, 0.5, 10],
    np.array([0.3, 0.5, 10])
])
def test_nonlin_power_halomod(k):
    a = 0.8
    pk = ccl.nonlin_matter_power(COSMO_HM, k, a)

    # New implementation
    mdef = ccl.halos.MassDef('vir', 'matter')
    hmf = ccl.halos.MassFuncSheth99(COSMO_HM, mdef,
                                    mass_def_strict=False,
                                    use_delta_c_fit=True)
    hbf = ccl.halos.HaloBiasSheth99(COSMO_HM, mass_def=mdef,
                                    mass_def_strict=False)
    cc = ccl.halos.ConcentrationDuffy08(mdef)
    prf = ccl.halos.HaloProfileNFW(cc)
    hmc = ccl.halos.HMCalculator(COSMO_HM, hmf, hbf, mdef)
    pkb = ccl.halos.halomod_power_spectrum(COSMO_HM, hmc, k, a,
                                           prf, normprof1=True)

    assert np.allclose(pk, pkb)
    assert np.all(np.isfinite(pk))
    assert np.shape(pk) == np.shape(k)


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
    # ('boltzmann_class', 'emu', 0.06), - this case is slow and not needed
    (None, 'emu', 0.06),
    ('bbks', 'emu', 0.06),
    ('eisenstein_hu', 'emu', 0.06),
])
def test_transfer_matter_power_nu_raises(tf, pk, m_nu):
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function=tf, matter_power_spectrum=pk, m_nu=m_nu)

    if tf is not None:
        with pytest.warns(CCLWarning):
            ccl.linear_matter_power(cosmo, 1, 1)

    with pytest.raises(CCLError):
        ccl.nonlin_matter_power(cosmo, 1, 1)


@pytest.mark.parametrize('tf', [
    'boltzmann_class', 'boltzmann_camb'])
def test_power_sigma8norm_norms_consistent(tf):
    # make a cosmo with A_s
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2e-9, n_s=0.96,
        transfer_function=tf)
    sigma8 = ccl.sigma8(cosmo)

    # remake same but now give sigma8
    cosmo_s8 = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=sigma8, n_s=0.96,
        transfer_function=tf)

    # make sure they come out the same-ish
    assert np.allclose(ccl.sigma8(cosmo), ccl.sigma8(cosmo_s8))

    # and that the power spectra look right
    a = 0.8
    gfac = (
        ccl.growth_factor(cosmo, a) / ccl.growth_factor(cosmo_s8, a))**2
    pk_rat = (
        ccl.linear_matter_power(cosmo, 1e-4, a) /
        ccl.linear_matter_power(cosmo_s8, 1e-4, a))
    assert np.allclose(pk_rat, gfac)
