import numpy as np
import pytest
import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
M200 = ccl.halos.MassDef200c()


def one_f(cosmo, M, a=None, mdef=None):
    if np.ndim(M) == 0:
        return 1
    else:
        return np.ones(M.size)


def smoke_assert_prof_real(profile):
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
        p = profile._real(COSMO, r, m, 1., M200)
        assert np.shape(p) == sh


def test_defaults():
    p = ccl.halos.HaloProfile()
    with pytest.raises(NotImplementedError):
        p.real(None, None, None, None)
    with pytest.raises(NotImplementedError):
        p.fourier(None, None, None, None)


@pytest.mark.parametrize('prof_class',
                         [ccl.halos.HaloProfileNFW,
                          ccl.halos.HaloProfileHernquist,
                          ccl.halos.HaloProfileEinasto])
def test_empirical_smoke(prof_class):
    c = ccl.halos.ConcentrationDuffy08(M200)
    with pytest.raises(TypeError):
        p = prof_class(None)

    if prof_class == ccl.halos.HaloProfileNFW:
        with pytest.raises(ValueError):
            p = prof_class(c,
                           projected_analytic=True,
                           truncated=True)

    p = prof_class(c)
    smoke_assert_prof_real(p)


@pytest.mark.parametrize('prof_class',
                         [ccl.halos.HaloProfileGaussian,
                          ccl.halos.HaloProfilePowerLaw])
def test_simple_smoke(prof_class):
    def r_s(cosmo, M, a, mdef):
        return mdef.get_radius(cosmo, M, a)

    p = prof_class(r_s, one_f)
    smoke_assert_prof_real(p)


def test_gaussian_accuracy():
    def fk(k):
        return np.pi**1.5 * np.exp(-k**2 / 4)

    p = ccl.halos.HaloProfileGaussian(one_f, one_f)

    k_arr = np.logspace(-3, 2, 1024)
    fk_arr = p.fourier(COSMO, k_arr, 1., 1.)
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

    p = ccl.halos.HaloProfilePowerLaw(one_f, alpha_f)
    p.update_precision_fftlog(plaw_index=alpha)

    rt_arr = np.logspace(-3, 2, 1024)
    srt_arr = p.projected(COSMO, rt_arr, 1., 1.)
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

    p = ccl.halos.HaloProfilePowerLaw(one_f, alpha_f)
    p.update_precision_fftlog(plaw_index=alpha)

    k_arr = np.logspace(-3, 2, 1024)
    fk_arr = p.fourier(COSMO, k_arr, 1., 1.)
    fk_arr_pred = fk(k_arr)
    res = np.fabs(fk_arr / fk_arr_pred - 1)
    assert np.all(res < 5E-3)


@pytest.mark.parametrize("use_analytic", [True, False])
def test_nfw_accuracy(use_analytic):
    from scipy.special import sici

    tol = 1E-10 if use_analytic else 5E-3
    cM = ccl.halos.ConcentrationDuffy08(M200)
    p = ccl.halos.HaloProfileNFW(cM, fourier_analytic=use_analytic)
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
    fk_arr = p.fourier(COSMO, k_arr, M, a, M200)
    fk_arr_pred = fk(k_arr)
    # Normalize to  1 at low k
    res = np.fabs((fk_arr - fk_arr_pred) / M)
    assert np.all(res < tol)


def test_f2r():
    cM = ccl.halos.ConcentrationDuffy08(M200)
    p1 = ccl.halos.HaloProfileNFW(cM)
    # We force p2 to compute the real-space profile
    # by FFT-ing the Fourier-space one.
    p2 = ccl.halos.HaloProfileNFW(cM, fourier_analytic=True)
    p2._real = None
    p2.update_precision_fftlog(padding_hi_fftlog=1E3)

    M = 1E14
    a = 0.5
    r_Delta = M200.get_radius(COSMO, M, a) / a

    r_arr = np.logspace(-2, 1, ) * r_Delta
    pr_1 = p1.real(COSMO, r_arr, M, a, M200)
    pr_2 = p2.real(COSMO, r_arr, M, a, M200)

    id_good = r_arr < r_Delta  # Profile is 0 otherwise
    res = np.fabs(pr_2[id_good] / pr_1[id_good] - 1)
    assert np.all(res < 0.01)


@pytest.mark.parametrize('fourier_analytic', [True, False])
def test_nfw_projected_accuracy(fourier_analytic):
    cM = ccl.halos.ConcentrationDuffy08(M200)
    # Analytic projected profile
    p1 = ccl.halos.HaloProfileNFW(cM, truncated=False,
                                  projected_analytic=True)
    # FFTLog
    p2 = ccl.halos.HaloProfileNFW(cM, truncated=False,
                                  fourier_analytic=fourier_analytic)

    M = 1E14
    a = 0.5
    rt = np.logspace(-3, 2, 1024)
    srt1 = p1.projected(COSMO, rt, M, a, M200)
    srt2 = p2.projected(COSMO, rt, M, a, M200)

    res2 = np.fabs(srt2/srt1-1)
    assert np.all(res2 < 5E-3)


@pytest.mark.parametrize('fourier_analytic', [True, False])
def test_nfw_cumul2d_accuracy(fourier_analytic):
    cM = ccl.halos.ConcentrationDuffy08(M200)
    # Analytic cumul2d profile
    p1 = ccl.halos.HaloProfileNFW(cM, truncated=False,
                                  cumul2d_analytic=True)
    # FFTLog
    p2 = ccl.halos.HaloProfileNFW(cM, truncated=False,
                                  fourier_analytic=fourier_analytic)

    M = 1E14
    a = 1.0
    rt = np.logspace(-3, 2, 1024)
    srt1 = p1.cumul2d(COSMO, rt, M, a, M200)
    srt2 = p2.cumul2d(COSMO, rt, M, a, M200)

    res2 = np.fabs(srt2/srt1-1)
    assert np.all(res2 < 5E-3)
