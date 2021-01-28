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
    from pyccl.pyutils import assert_warns
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
    pk_old = assert_warns(
        ccl.CCLWarning,
        ccl.nonlin_matter_power, COSMO_HM, 1., 0.8)
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


@pytest.mark.parametrize('A', [
    1,
    1.0,
    [0.3, 0.5, 1],
    np.array([0.3, 0.5, 1])])
def test_kNL(A):
    knl = ccl.kNL(COSMO, A)
    assert np.all(np.isfinite(knl))
    assert np.shape(knl) == np.shape(A)


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
    'boltzmann_class', 'boltzmann_camb', 'boltzmann_isitgr'])
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


def test_input_lin_power_spectrum():
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7, n_s=0.965,
                          A_s=2e-9)
    a_arr = np.linspace(0.1, 1.0, 50)
    k_arr = np.logspace(np.log10(2e-4), np.log10(1), 1000)
    pk_arr = np.empty(shape=(len(a_arr), len(k_arr)))
    for i, a in enumerate(a_arr):
        pk_arr[i] = ccl.power.linear_matter_power(cosmo, k_arr, a)

    chi_from_ccl = ccl.background.comoving_radial_distance(cosmo, a_arr)
    hoh0_from_ccl = ccl.background.h_over_h0(cosmo, a_arr)
    growth_from_ccl = ccl.background.growth_factor(cosmo, a_arr)
    fgrowth_from_ccl = ccl.background.growth_rate(cosmo, a_arr)

    cosmo_input = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7, n_s=0.965,
                                A_s=2e-9)
    cosmo_input._set_background_from_arrays(a_array=a_arr,
                                            chi_array=chi_from_ccl,
                                            hoh0_array=hoh0_from_ccl,
                                            growth_array=growth_from_ccl,
                                            fgrowth_array=fgrowth_from_ccl)
    cosmo_input._set_linear_power_from_arrays(a_arr, k_arr, pk_arr)

    pk_CCL_input = ccl.power.linear_matter_power(cosmo_input, k_arr, 0.5)
    pk_CCL = ccl.power.linear_matter_power(cosmo, k_arr, 0.5)

    assert np.allclose(pk_CCL_input, pk_CCL, atol=0., rtol=1e-5)


def test_input_linpower_raises():
    cosmo_input = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7, n_s=0.965,
                                A_s=2e-9)
    with pytest.raises(ValueError):
        cosmo_input._set_linear_power_from_arrays()
    with pytest.raises(ValueError):
        cosmo_input.compute_linear_power()
        cosmo_input._set_linear_power_from_arrays()
    cosmo_input = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7, n_s=0.965,
                                A_s=2e-9)
    with pytest.raises(ValueError):
        cosmo_input._compute_linear_power_from_arrays()


def test_input_nonlin_power_spectrum():
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7, n_s=0.965,

                          A_s=2e-9)
    a_arr = np.linspace(0.1, 1.0, 50)
    k_arr = np.logspace(np.log10(2e-4), np.log10(1), 1000)
    pk_arr = np.empty(shape=(len(a_arr), len(k_arr)))
    for i, a in enumerate(a_arr):
        pk_arr[i] = ccl.power.nonlin_matter_power(cosmo, k_arr, a)

    chi_from_ccl = ccl.background.comoving_radial_distance(cosmo, a_arr)
    hoh0_from_ccl = ccl.background.h_over_h0(cosmo, a_arr)
    growth_from_ccl = ccl.background.growth_factor(cosmo, a_arr)
    fgrowth_from_ccl = ccl.background.growth_rate(cosmo, a_arr)

    cosmo_input = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7, n_s=0.965,
                                A_s=2e-9,)
    cosmo_input._set_background_from_arrays(a_array=a_arr,
                                            chi_array=chi_from_ccl,
                                            hoh0_array=hoh0_from_ccl,
                                            growth_array=growth_from_ccl,
                                            fgrowth_array=fgrowth_from_ccl)
    cosmo_input._set_nonlin_power_from_arrays(a_arr, k_arr, pk_arr)

    pk_CCL_input = ccl.power.nonlin_matter_power(cosmo_input, k_arr, 0.5)
    pk_CCL = ccl.power.nonlin_matter_power(cosmo, k_arr, 0.5)

    assert np.allclose(pk_CCL_input, pk_CCL, atol=0., rtol=1e-5)


def test_input_nonlin_raises():
    cosmo_input = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7, n_s=0.965,
                                A_s=2e-9)
    with pytest.raises(ValueError):
        cosmo_input._set_nonlin_power_from_arrays()
    with pytest.raises(ValueError):
        cosmo_input.compute_nonlin_power()
        cosmo_input._set_nonlin_power_from_arrays()
    cosmo_input = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7, n_s=0.965,
                                A_s=2e-9)
    with pytest.raises(ValueError):
        cosmo_input._compute_nonlin_power_from_arrays()
