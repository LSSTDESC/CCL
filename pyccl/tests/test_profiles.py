import numpy as np
import pytest
import pyccl as ccl
from pyccl import UnlockInstance


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
M200 = ccl.halos.MassDef200c()
M500c = ccl.halos.MassDef(500, 'critical')


def one_f(cosmo, M, a=1, mass_def=M200):
    if np.ndim(M) == 0:
        return 1
    else:
        return np.ones(M.size)


def smoke_assert_prof_real(profile, method='_real'):
    sizes = [(0, 0),
             (2, 0),
             (0, 2),
             (2, 3),
             (1, 3),
             (3, 1)]
    shapes = [(),
              (2,),
              (2,),
              (2, 3),
              (1, 3),
              (3, 1)]
    for (sm, sr), sh in zip(sizes, shapes):
        if sr == 0:
            r = 0.5
        else:
            # Don't include 0 to avoid 1/0 at origin.
            r = np.linspace(0.001, 1., sr)
        if sm == 0:
            m = 1E12
        else:
            m = np.geomspace(1E10, 1E14, sm)
        p = getattr(profile, method)(COSMO, r, m, 1., mass_def=M200)
        assert np.shape(p) == sh


# TODO: Uncomment once the profiles can be compared.
# def test_profiles_equal():
#     M200m = ccl.halos.MassDef200m()
#     cm = ccl.halos.ConcentrationDuffy08(mass_def=M200m)
#     p1 = ccl.halos.HaloProfileHOD(c_m_relation=cm, lMmin_0=12.)

#     # different profile types
#     p2 = ccl.halos.HaloProfilePressureGNFW()
#     assert p1 != p2

#     # equal profiles
#     p2 = p1
#     assert p1 == p2

#     # equivalent profiles
#     cm2 = ccl.halos.ConcentrationDuffy08(mass_def=M200m)
#     p2 = ccl.halos.HaloProfileHOD(c_m_relation=cm2, lMmin_0=12.)
#     assert p1 == p2

#     # different parameters
#     p2 = ccl.halos.HaloProfileHOD(c_m_relation=cm, lMmin_0=11.)
#     assert p1 != p2

#     # different mass-concentration
#     cm2 = ccl.halos.ConcentrationConstant()
#     p2 = ccl.halos.HaloProfileHOD(c_m_relation=cm2, lMmin_0=12.)
#     assert p1 != p2

#     # different mass-concentration mass definition
#     M200c = ccl.halos.MassDef200c()
#     cm2 = ccl.halos.ConcentrationDuffy08(mass_def=M200c)
#     p2 = ccl.halos.HaloProfileHOD(c_m_relation=cm2, lMmin_0=12.)
#     assert p1 != p2

#     # different FFTLog
#     p2 = ccl.halos.HaloProfileHOD(c_m_relation=cm, lMmin_0=12.)
#     p2.update_precision_fftlog(**{"plaw_fourier": -2.0})
#     assert p1 != p2


@pytest.mark.parametrize('prof_class',
                         [ccl.halos.HaloProfileNFW,
                          ccl.halos.HaloProfileHernquist,
                          ccl.halos.HaloProfileEinasto])
def test_empirical_smoke(prof_class):
    c = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    with pytest.raises(TypeError):
        p = prof_class(c_m_relation=None)

    if prof_class in [ccl.halos.HaloProfileNFW,
                      ccl.halos.HaloProfileHernquist]:
        with pytest.raises(ValueError):
            p = prof_class(c_m_relation=c,
                           projected_analytic=True,
                           truncated=True)
        with pytest.raises(ValueError):
            p = prof_class(c_m_relation=c,
                           cumul2d_analytic=True,
                           truncated=True)
        p = prof_class(c_m_relation=c)
        smoke_assert_prof_real(p, method='_fourier_analytic')
        smoke_assert_prof_real(p, method='_projected_analytic')
        smoke_assert_prof_real(p, method='_cumul2d_analytic')

    p = prof_class(c_m_relation=c)
    smoke_assert_prof_real(p, method='real')
    smoke_assert_prof_real(p, method='projected')
    smoke_assert_prof_real(p, method='fourier')


def test_empirical_smoke_CIB():
    with pytest.raises(TypeError):
        ccl.halos.HaloProfileCIBShang12(c_m_relation=None, nu_GHz=417)


def test_cib_smoke():
    c = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    p = ccl.halos.HaloProfileCIBShang12(c_m_relation=c, nu_GHz=217)
    beta_old = p.beta
    smoke_assert_prof_real(p, method='_real')
    smoke_assert_prof_real(p, method='_fourier')
    smoke_assert_prof_real(p, method='_fourier_variance')
    p.update_parameters(alpha=1.24, T0=20.0)
    assert p.alpha == 1.24
    assert p.T0 == 20.0
    assert p.beta == beta_old
    for n in ['alpha', 'T0', 'beta', 'gamma',
              's_z', 'Mmin', 'L0', 'sigLM']:
        p.update_parameters(**{n: 1234.})
        assert getattr(p, n) == 1234.


def test_cib_raises():
    with pytest.raises(TypeError):
        ccl.halos.HaloProfileCIBShang12(c_m_relation="my_conc", nu_GHz=217)


def test_cib_2pt_raises():
    c = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    p_cib = ccl.halos.HaloProfileCIBShang12(c_m_relation=c, nu_GHz=217)
    p_tSZ = ccl.halos.HaloProfilePressureGNFW()
    p2pt = ccl.halos.Profile2ptCIB()
    with pytest.raises(TypeError):
        p2pt.fourier_2pt(COSMO, 0.1, 1E13, 1., p_tSZ,
                         mass_def=M200)
    with pytest.raises(TypeError):
        p2pt.fourier_2pt(COSMO, 0.1, 1E13, 1., p_cib,
                         prof2=p_tSZ, mass_def=M200)


def test_einasto_smoke():
    c = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    p = ccl.halos.HaloProfileEinasto(c_m_relation=c)
    for M in [1E14, [1E14, 1E15]]:
        alpha_from_cosmo = p._get_alpha(COSMO, M, 1., M200)

        p.update_parameters(alpha=1.)
        alpha = p._get_alpha(COSMO, M, 1., M200)
        assert np.ndim(M) == np.ndim(alpha)
        assert np.all(p._get_alpha(COSMO, M, 1., M200) == np.full_like(M, 1.))

        p.update_parameters(alpha='cosmo')
        assert np.ndim(M) == np.ndim(alpha_from_cosmo)
        assert np.all(p._get_alpha(COSMO, M, 1., M200) == alpha_from_cosmo)


def test_gnfw_smoke():
    p = ccl.halos.HaloProfilePressureGNFW()
    beta_old = p.beta
    smoke_assert_prof_real(p)
    p.update_parameters(mass_bias=0.7,
                        alpha=1.24)
    assert p.alpha == 1.24
    assert p.mass_bias == 0.7
    assert p.beta == beta_old
    for n in ['P0', 'P0_hexp', 'alpha',
              'beta', 'gamma', 'alpha_P',
              'c500', 'mass_bias', 'x_out']:
        p.update_parameters(**{n: 1.314159})
        assert getattr(p, n) == 1.314159


def test_gnfw_refourier():
    p = ccl.halos.HaloProfilePressureGNFW()
    # Create Fourier template
    p._integ_interp()
    p_f1 = p.fourier(COSMO, 1., 1E13, 1., mass_def=M500c)
    # Check the Fourier profile gets recalculated
    p.update_parameters(alpha=1.32, c500=p.c500+0.1)
    p_f2 = p.fourier(COSMO, 1., 1E13, 1., mass_def=M500c)
    assert p_f1 != p_f2


def test_hod_smoke():
    prof_class = ccl.halos.HaloProfileHOD
    c = ccl.halos.ConcentrationDuffy08(mass_def=M200)

    with pytest.raises(TypeError):
        p = prof_class(c_m_relation=None)

    p = prof_class(c_m_relation=c)
    smoke_assert_prof_real(p)
    smoke_assert_prof_real(p, method='_usat_real')
    smoke_assert_prof_real(p, method='_usat_fourier')
    smoke_assert_prof_real(p, method='_fourier')
    smoke_assert_prof_real(p, method='_fourier_variance')
    for n in ['lMmin_0', 'lMmin_p', 'lM0_0', 'lM0_p',
              'lM1_0', 'lM1_p', 'siglM_0', 'siglM_p',
              'fc_0', 'fc_p', 'alpha_0', 'alpha_p',
              'bg_0', 'bg_p', 'bmax_0', 'bmax_p',
              'a_pivot']:
        p.update_parameters(**{n: 1234.})
        assert getattr(p, n) == 1234.


@pytest.mark.parametrize('real_prof', [True, False])
def test_hod_ns_independent(real_prof):
    def func(prof):
        return prof._real if real_prof else prof._fourier

    c = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    hmd = c.mass_def
    p1 = ccl.halos.HaloProfileHOD(c_m_relation=c,
                                  lMmin_0=12.,
                                  ns_independent=False)
    p2 = ccl.halos.HaloProfileHOD(c_m_relation=c,
                                  lMmin_0=12.,
                                  ns_independent=True)
    # M < Mmin
    f1 = func(p1)(COSMO, 0.01, 1e10, 1., mass_def=hmd)
    assert np.all(f1 == 0)
    f2 = func(p2)(COSMO, 0.01, 1e10, 1., mass_def=hmd)
    assert np.all(f2 > 0)
    # M > Mmin
    f1 = func(p1)(COSMO, 0.01, 1e14, 1., mass_def=hmd)
    f2 = func(p2)(COSMO, 0.01, 1e14, 1., mass_def=hmd)
    assert np.allclose(f1, f2, rtol=0)
    # M == Mmin
    f1 = func(p1)(COSMO, 0.01, 1e12, 1., mass_def=hmd)
    f2 = func(p2)(COSMO, 0.01, 1e12, 1., mass_def=hmd)
    assert np.allclose(2*f1, f2+0.5, rtol=0)

    if not real_prof:
        f1 = p1._fourier_variance(COSMO, 0.01, 1e10, 1., mass_def=hmd)
        f2 = p2._fourier_variance(COSMO, 0.01, 1e10, 1., mass_def=hmd)
        assert f2 > f1 == 0

    p1.update_parameters(ns_independent=True)
    assert p1.ns_independent is True


def test_hod_2pt():
    pbad = ccl.halos.HaloProfilePressureGNFW()
    c = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    pgood = ccl.halos.HaloProfileHOD(c_m_relation=c)
    pgood_b = ccl.halos.HaloProfileHOD(c_m_relation=c)
    p2 = ccl.halos.Profile2ptHOD()
    F0 = p2.fourier_2pt(COSMO, 1., 1e13, 1., pgood,
                        prof2=pgood,
                        mass_def=M200)
    assert np.allclose(p2.fourier_2pt(COSMO, 1., 1e13, 1., pgood,
                                      prof2=None, mass_def=M200),
                       F0, rtol=0)

    with pytest.raises(TypeError):
        p2.fourier_2pt(COSMO, 1., 1E13, 1., pbad,
                       mass_def=M200)

    # TODO: bring back when proper __eq__s are implemented
    # doesn't raise because profiles are equivalent
    # p2.fourier_2pt(COSMO, 1., 1E13, 1., pgood,
    #                prof2=pgood, mass_def=M200)

    with pytest.raises(ValueError):
        pgood_b.update_parameters(lM0_0=10.)
        p2.fourier_2pt(COSMO, 1., 1E13, 1., pgood,
                       prof2=pgood_b, mass_def=M200)


def test_2pt_rcorr_smoke():
    c = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    p = ccl.halos.HaloProfileNFW(c_m_relation=c)
    F0 = ccl.halos.Profile2pt().fourier_2pt(COSMO, 1., 1e13, 1., p,
                                            mass_def=M200)
    p2pt = ccl.halos.Profile2pt(r_corr=0)
    F1 = p2pt.fourier_2pt(COSMO, 1., 1e13, 1., p, mass_def=M200)
    assert F0 == F1
    F2 = p2pt.fourier_2pt(COSMO, 1., 1e13, 1., p, prof2=p, mass_def=M200)
    assert F1 == F2

    p2pt.update_parameters(r_corr=-1.)
    assert p2pt.r_corr == -1.
    F3 = p2pt.fourier_2pt(COSMO, 1., 1e13, 1., p, mass_def=M200)
    assert F3 == 0

    # Errors
    with pytest.raises(TypeError):
        p2pt.fourier_2pt(COSMO, 1., 1e13, 1., None, mass_def=M200)
    with pytest.raises(TypeError):
        p2pt.fourier_2pt(COSMO, 1., 1e13, 1., p, prof2=0, mass_def=M200)


@pytest.mark.parametrize('prof_class',
                         [ccl.halos.HaloProfileGaussian,
                          ccl.halos.HaloProfilePowerLaw])
def test_simple_smoke(prof_class):
    def r_s(cosmo, M, a, mass_def):
        return mass_def.get_radius(cosmo, M, a)

    if prof_class == ccl.halos.HaloProfileGaussian:
        p = prof_class(r_scale=r_s, rho0=one_f)
    else:
        p = prof_class(r_scale=r_s, tilt=one_f)
    smoke_assert_prof_real(p)


def test_gaussian_accuracy():
    def fk(k):
        return np.pi**1.5 * np.exp(-k**2 / 4)

    p = ccl.halos.HaloProfileGaussian(r_scale=one_f, rho0=one_f)

    k_arr = np.logspace(-3, 2, 1024)
    fk_arr = p.fourier(COSMO, k_arr, 1., 1., mass_def=M200)
    fk_arr_pred = fk(k_arr)
    res = np.fabs(fk_arr - fk_arr_pred)
    assert np.all(res < 5E-3)


@pytest.mark.parametrize('alpha', [-1.2, -2., -2.8])
def test_projected_plaw_accuracy(alpha):
    from scipy.special import gamma

    prefac = (np.pi**0.5 * gamma(-(alpha + 1) / 2) /
              gamma(-alpha / 2))

    def s_r_t(rt):
        return prefac * rt**(1 + alpha)

    def alpha_f(cosmo, a):
        return alpha

    p = ccl.halos.HaloProfilePowerLaw(r_scale=one_f, tilt=alpha_f)
    p.update_precision_fftlog(plaw_fourier=alpha)

    rt_arr = np.logspace(-3, 2, 1024)
    srt_arr = p.projected(COSMO, rt_arr, 1., 1., mass_def=M200)
    srt_arr_pred = s_r_t(rt_arr)
    res = np.fabs(srt_arr / srt_arr_pred - 1)
    assert np.all(res < 5E-3)


@pytest.mark.parametrize('alpha', [-1.2, -2., -2.8])
def test_plaw_accuracy(alpha):
    from scipy.special import gamma

    prefac = (2.**(3+alpha) * np.pi**1.5 *
              gamma((3 + alpha) / 2) /
              gamma(-alpha / 2))

    def fk(k):
        return prefac / k**(3 + alpha)

    def alpha_f(cosmo, a):
        return alpha

    p = ccl.halos.HaloProfilePowerLaw(r_scale=one_f, tilt=alpha_f)
    p.update_precision_fftlog(plaw_fourier=alpha)

    k_arr = np.logspace(-3, 2, 1024)
    fk_arr = p.fourier(COSMO, k_arr, 1., 1., mass_def=M200)
    fk_arr_pred = fk(k_arr)
    res = np.fabs(fk_arr / fk_arr_pred - 1)
    assert np.all(res < 5E-3)


def get_nfw(real=False, fourier=False):
    # Return a subclass of the NFW profile with or without fourier analytic.
    NFW = type("NFW", (ccl.halos.HaloProfileNFW,), {})
    if real:
        NFW._real = ccl.halos.HaloProfileNFW._real
    if fourier:
        NFW._fourier = ccl.halos.HaloProfileNFW._fourier_analytic
    return NFW


@pytest.mark.parametrize("use_analytic", [True, False])
def test_nfw_accuracy(use_analytic):
    from scipy.special import sici

    tol = 1E-10 if use_analytic else 5E-3
    cM = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    p = get_nfw(real=True, fourier=use_analytic)(c_m_relation=cM)
    M = 1E14
    a = 0.5
    c = cM.get_concentration(COSMO, M, a)
    r_Delta = M200.get_radius(COSMO, M, a) / a
    r_s = r_Delta / c

    def fk(k):
        x = k * r_s
        Si1, Ci1 = sici((1 + c) * x)
        Si2, Ci2 = sici(x)
        P1 = 1 / (np.log(1+c) - c / (1 + c))
        P2 = np.sin(x) * (Si1 - Si2) + np.cos(x) * (Ci1 - Ci2)
        P3 = np.sin(c * x)/((1 + c) * x)
        return M * P1 * (P2 - P3)

    k_arr = np.logspace(-2, 2, 256) / r_Delta
    fk_arr = p.fourier(COSMO, k_arr, M, a, mass_def=M200)
    fk_arr_pred = fk(k_arr)
    # Normalize to  1 at low k
    res = np.fabs((fk_arr - fk_arr_pred) / M)
    assert np.all(res < tol)


def test_nfw_f2r():
    cM = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    p1 = ccl.halos.HaloProfileNFW(c_m_relation=cM)
    # We force p2 to compute the real-space profile
    # by FFT-ing the Fourier-space one.
    p2 = get_nfw(fourier=True)(c_m_relation=cM)
    p2.update_precision_fftlog(padding_hi_fftlog=1E3)

    M = 1E14
    a = 0.5
    r_Delta = M200.get_radius(COSMO, M, a) / a

    r_arr = np.logspace(-2, 1, ) * r_Delta
    pr_1 = p1.real(COSMO, r_arr, M, a, mass_def=M200)
    pr_2 = p2.real(COSMO, r_arr, M, a, mass_def=M200)

    id_good = r_arr < r_Delta  # Profile is 0 otherwise
    res = np.fabs(pr_2[id_good] / pr_1[id_good] - 1)
    assert np.all(res < 0.01)


@pytest.mark.parametrize('fourier_analytic', [True, False])
def test_nfw_projected_accuracy(fourier_analytic):
    cM = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    # Analytic projected profile
    p1 = ccl.halos.HaloProfileNFW(c_m_relation=cM, truncated=False,
                                  projected_analytic=True)
    # FFTLog
    p2 = get_nfw(fourier=fourier_analytic)(c_m_relation=cM, truncated=False)

    M = 1E14
    a = 0.5
    rt = np.logspace(-3, 2, 1024)
    srt1 = p1.projected(COSMO, rt, M, a, mass_def=M200)
    srt2 = p2.projected(COSMO, rt, M, a, mass_def=M200)

    res2 = np.fabs(srt2/srt1-1)
    assert np.all(res2 < 5E-3)


@pytest.mark.parametrize('fourier_analytic', [True, False])
def test_nfw_cumul2d_accuracy(fourier_analytic):
    cM = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    # Analytic cumul2d profile
    p1 = ccl.halos.HaloProfileNFW(c_m_relation=cM, truncated=False,
                                  cumul2d_analytic=True)
    # FFTLog
    p2 = get_nfw(fourier=fourier_analytic)(c_m_relation=cM, truncated=False)

    M = 1E14
    a = 1.0
    rt = np.logspace(-3, 2, 1024)
    srt1 = p1.cumul2d(COSMO, rt, M, a, mass_def=M200)
    srt2 = p2.cumul2d(COSMO, rt, M, a, mass_def=M200)

    res2 = np.fabs(srt2/srt1-1)
    assert np.all(res2 < 5E-3)


def test_upd_fftlog_raises():
    prof = ccl.halos.HaloProfilePressureGNFW()
    new_params = {"hello_there": 0.}
    with pytest.raises(KeyError):
        prof.update_precision_fftlog(**new_params)


@pytest.mark.parametrize("use_analytic", [True, False])
def test_hernquist_accuracy(use_analytic):
    from scipy.special import sici

    tol = 1E-10 if use_analytic else 5E-3
    cM = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    p = ccl.halos.HaloProfileHernquist(c_m_relation=cM,
                                       fourier_analytic=use_analytic)
    M = 1E14
    a = 0.5
    c = cM.get_concentration(COSMO, M, a)
    r_Delta = M200.get_radius(COSMO, M, a) / a
    r_s = r_Delta / c

    def fk(k):
        x = k * r_s
        cp1 = c + 1
        Si1, Ci1 = sici(cp1 * x)
        Si2, Ci2 = sici(x)
        P1 = 1 / ((c / cp1)**2 / 2)
        P2 = x * np.sin(x) * (Ci1 - Ci2) - x * np.cos(x) * (Si1 - Si2)
        P3 = (-1 + np.sin(c * x) / (cp1**2 * x)
              + cp1 * np.cos(c * x) / (cp1**2))
        return M * P1 * (P2 - P3) / 2

    k_arr = np.logspace(-2, 2, 256) / r_Delta
    fk_arr = p.fourier(COSMO, k_arr, M, a, mass_def=M200)
    fk_arr_pred = fk(k_arr)
    # Normalize to  1 at low k
    res = np.fabs((fk_arr - fk_arr_pred) / M)
    assert np.all(res < tol)


def test_hernquist_f2r():
    cM = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    p1 = ccl.halos.HaloProfileHernquist(c_m_relation=cM)
    # We force p2 to compute the real-space profile
    # by FFT-ing the Fourier-space one.
    p2 = ccl.halos.HaloProfileHernquist(c_m_relation=cM, fourier_analytic=True)
    with UnlockInstance(p2):
        p2._real = None
    p2.update_precision_fftlog(padding_hi_fftlog=1E3)

    M = 1E14
    a = 0.5
    r_Delta = M200.get_radius(COSMO, M, a) / a

    r_arr = np.logspace(-2, 1, ) * r_Delta
    pr_1 = p1.real(COSMO, r_arr, M, a, mass_def=M200)
    pr_2 = p2.real(COSMO, r_arr, M, a, mass_def=M200)

    id_good = r_arr < r_Delta  # Profile is 0 otherwise
    res = np.fabs(pr_2[id_good] / pr_1[id_good] - 1)
    assert np.all(res < 0.01)


@pytest.mark.parametrize('fourier_analytic', [True, False])
def test_hernquist_projected_accuracy(fourier_analytic):
    cM = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    # Analytic projected profile
    p1 = ccl.halos.HaloProfileHernquist(c_m_relation=cM, truncated=False,
                                        projected_analytic=True)
    # FFTLog
    p2 = ccl.halos.HaloProfileHernquist(c_m_relation=cM, truncated=False,
                                        fourier_analytic=fourier_analytic)

    M = 1E14
    a = 0.5
    rt = np.logspace(-3, 2, 1024)
    srt1 = p1.projected(COSMO, rt, M, a, mass_def=M200)
    srt2 = p2.projected(COSMO, rt, M, a, mass_def=M200)

    res2 = np.fabs(srt2/srt1-1)
    assert np.all(res2 < 5E-3)


@pytest.mark.parametrize('fourier_analytic', [True, False])
def test_hernquist_cumul2d_accuracy(fourier_analytic):
    cM = ccl.halos.ConcentrationDuffy08(mass_def=M200)
    # Analytic cumul2d profile
    p1 = ccl.halos.HaloProfileHernquist(c_m_relation=cM, truncated=False,
                                        cumul2d_analytic=True)
    # FFTLog
    p2 = ccl.halos.HaloProfileHernquist(c_m_relation=cM, truncated=False,
                                        fourier_analytic=fourier_analytic)
    M = 1E14
    a = 1.0
    rt = np.logspace(-3, 2, 1024)
    srt1 = p1.cumul2d(COSMO, rt, M, a, mass_def=M200)
    srt2 = p2.cumul2d(COSMO, rt, M, a, mass_def=M200)

    res2 = np.fabs(srt2/srt1-1)
    assert np.all(res2 < 5E-3)
