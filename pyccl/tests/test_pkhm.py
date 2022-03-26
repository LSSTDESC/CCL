import numpy as np
import pytest
from . import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
M200 = ccl.halos.MassDef200m()
HMF = ccl.halos.MassFuncTinker10(mass_def=M200)
HBF = ccl.halos.HaloBiasTinker10(mass_def=M200)
CON = ccl.halos.ConcentrationDuffy08(mass_def=M200)
P1 = ccl.halos.HaloProfileNFW(c_m_relation=CON, fourier_analytic=True)
P2 = P1
PKC = ccl.halos.Profile2pt()
KK = np.geomspace(1E-3, 10, 32)
MM = np.geomspace(1E11, 1E15, 16)
AA = 1.0
PK2D = ccl.Pk2D(cosmo=COSMO, pkfunc=lambda k, a: a / k)


def test_hmc_from_string_smoke():
    hmc0 = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                  mass_def=M200)
    # all strings
    hmc1 = ccl.halos.HMCalculator(mass_function="Tinker10",
                                  halo_bias="Tinker10",
                                  mass_def="200m")
    # no quality assurance of the following lines, because we
    # require an explicit type check, not an instance check (E721)
    assert type(hmc1.mass_function) == type(hmc0.mass_function)  # noqa
    assert type(hmc1.halo_bias) == type(hmc0.halo_bias)          # noqa
    assert type(hmc1.mass_def) == type(hmc0.mass_def)            # noqa

    # some strings
    hmc2 = ccl.halos.HMCalculator(mass_function=HMF, halo_bias="Tinker10",
                                  mass_def="vir")
    assert isinstance(hmc2.mass_def, ccl.halos.MassDefVir)


def test_hmc_raises():
    with pytest.raises(TypeError):
        ccl.halos.HMCalculator(mass_function=None, halo_bias=HBF,
                               mass_def=M200)
    with pytest.raises(TypeError):
        ccl.halos.HMCalculator(mass_function=HMF, halo_bias=None,
                               mass_def=M200)
    with pytest.raises(TypeError):
        ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                               mass_def=None)


def test_prof2pt_smoke():
    uk_NFW = P1.fourier(COSMO, KK, MM, AA,
                        mass_def=M200)
    uk_EIN = P2.fourier(COSMO, KK, MM, AA,
                        mass_def=M200)
    # Variance
    cv_NN = PKC.fourier_2pt(COSMO, KK, MM, AA, P1,
                            mass_def=M200)
    assert np.all(np.fabs((cv_NN - uk_NFW**2)) < 1E-10)

    # 2-point
    cv_NE = PKC.fourier_2pt(COSMO, KK, MM, AA, P1,
                            prof2=P2, mass_def=M200)
    assert np.all(np.fabs((cv_NE - uk_NFW * uk_EIN)) < 1E-10)


def test_prof2pt_errors():
    # Wrong first profile
    with pytest.raises(TypeError):
        PKC.fourier_2pt(COSMO, KK, MM, AA, None,
                        prof2=None, mass_def=M200)

    # Wrong second profile
    with pytest.raises(TypeError):
        PKC.fourier_2pt(COSMO, KK, MM, AA, P1,
                        prof2=M200, mass_def=M200)


def smoke_assert_pkhm_real(func):
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
    for (sa, sk), sh in zip(sizes, shapes):
        if sk == 0:
            k = 0.1
        else:
            k = np.logspace(-2., 0., sk)
        if sa == 0:
            a = 1.
        else:
            a = np.linspace(0.5, 1., sa)
        p = func(k, a)
        assert np.shape(p) == sh
        assert np.all(np.isfinite(p))


@pytest.mark.parametrize('norm', [True, False])
def test_pkhm_mean_profile_smoke(norm):
    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200, nlM=2)

    def f(k, a):
        return ccl.halos.halomod_mean_profile_1pt(COSMO, hmc, k, a,
                                                  P1, normprof=norm)
    smoke_assert_pkhm_real(f)


@pytest.mark.parametrize('norm', [True, False])
def test_pkhm_bias_smoke(norm):
    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200, nlM=2)

    def f(k, a):
        return ccl.halos.halomod_bias_1pt(COSMO, hmc, k, a,
                                          P1, normprof=norm)
    smoke_assert_pkhm_real(f)


@pytest.mark.parametrize('pars',
                         [{'cv': None, 'norm': True,
                           'pk': 'linear', 'h1': True,
                           'h2': True, 'itg': 'simpson',
                           'p2': None},
                          {'cv': PKC, 'norm': True,
                           'pk': 'linear', 'h1': True,
                           'h2': True, 'itg': 'simpson',
                           'p2': None},
                          {'cv': None, 'norm': False,
                           'pk': 'linear', 'h1': True,
                           'h2': True, 'itg': 'simpson',
                           'p2': None},
                          {'cv': None, 'norm': True,
                           'pk': 'nonlinear', 'h1': True,
                           'h2': True, 'itg': 'simpson',
                           'p2': None},
                          {'cv': None, 'norm': True,
                           'pk': PK2D, 'h1': True,
                           'h2': True, 'itg': 'simpson',
                           'p2': None},
                          {'cv': None, 'norm': True,
                           'pk': None, 'h1': True,
                           'h2': True, 'itg': 'simpson',
                           'p2': None},
                          {'cv': None, 'norm': True,
                           'pk': 'linear', 'h1': False,
                           'h2': True, 'itg': 'simpson',
                           'p2': None},
                          {'cv': None, 'norm': True,
                           'pk': 'linear', 'h1': True,
                           'h2': False, 'itg': 'simpson',
                           'p2': None},
                          {'cv': None, 'norm': True,
                           'pk': 'linear', 'h1': False,
                           'h2': False, 'itg': 'simpson',
                           'p2': None},
                          {'cv': None, 'norm': True,
                           'pk': 'linear', 'h1': True,
                           'h2': True, 'itg': 'spline',
                           'p2': None},
                          {'cv': None, 'norm': True,
                           'pk': 'linear', 'h1': True,
                           'h2': True, 'itg': 'simpson',
                           'p2': P2},
                          {'cv': None, 'norm': False,
                           'pk': 'linear', 'h1': True,
                           'h2': True, 'itg': 'simpson',
                           'p2': P2}])
def test_pkhm_pk_smoke(pars):
    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200, nlM=2)

    def f(k, a):
        return ccl.halos.halomod_power_spectrum(COSMO, hmc, k, a, P1,
                                                prof_2pt=pars['cv'],
                                                normprof=pars['norm'],
                                                normprof2=pars['norm'],
                                                p_of_k_a=pars['pk'],
                                                prof2=pars['p2'],
                                                get_1h=pars['h1'],
                                                get_2h=pars['h2'])
    smoke_assert_pkhm_real(f)


def test_pkhm_pk2d():
    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200)
    k_arr = KK
    a_arr = np.linspace(0.3, 1, 10)
    pk_arr = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr,
                                              P1, normprof=True,
                                              normprof2=True)

    # Input sampling
    pk2d = ccl.halos.halomod_Pk2D(COSMO, hmc, P1,
                                  lk_arr=np.log(k_arr),
                                  a_arr=a_arr, normprof=True)
    pk_arr_2 = np.array([pk2d.eval(k_arr, a, COSMO)
                         for a in a_arr])
    assert np.all(np.fabs((pk_arr / pk_arr_2 - 1)).flatten()
                  < 1E-4)

    # Standard sampling
    pk2d = ccl.halos.halomod_Pk2D(COSMO, hmc, P1,
                                  normprof=True)
    pk_arr_2 = np.array([pk2d.eval(k_arr, a, COSMO)
                         for a in a_arr])
    assert np.all(np.fabs((pk_arr / pk_arr_2 - 1)).flatten()
                  < 1E-4)

    # 1h/2h transition
    def alpha0(a):  # no smoothing
        return 1.

    def alpha1(a):
        return 0.7

    pk0 = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr, P1,
                                           normprof=True, normprof2=True,
                                           smooth_transition=None)
    pk1 = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr, P1,
                                           normprof=True, normprof2=True,
                                           smooth_transition=alpha0)
    assert np.allclose(pk0, pk1, rtol=0)
    pk2 = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr, P1,
                                           normprof=True, normprof2=True,
                                           smooth_transition=alpha1)
    assert np.all(pk2/pk0 > 1.)

    # 1-halo damping
    def ks0(a):  # no damping
        return 1e-16

    def ks1(a):  # fully suppressed
        return 1e16

    def ks2(a):  # reasonable
        return 0.04

    pk0 = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr, P1,
                                           normprof=True, normprof2=True,
                                           suppress_1h=None, get_2h=False)
    pk1 = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr, P1,
                                           normprof=True, normprof2=True,
                                           suppress_1h=ks0, get_2h=False)
    assert np.allclose(pk0, pk1, rtol=0)
    pk2 = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr, P1,
                                           normprof=True, normprof2=True,
                                           suppress_1h=ks1, get_2h=False)
    assert np.allclose(pk2, 0, rtol=0)
    pk3 = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr, P1,
                                           normprof=True, normprof2=True,
                                           suppress_1h=ks2, get_2h=False)
    fact = (k_arr/0.04)**4 / (1 + (k_arr/0.04)**4)
    assert np.allclose(pk3, pk0*fact, rtol=0)


def test_pkhm_errors():
    # Wrong integration
    with pytest.raises(NotImplementedError):
        ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                               mass_def=M200,
                               integration_method_M='Sampson')

    # Wrong hmf
    with pytest.raises(TypeError):
        ccl.halos.HMCalculator(mass_function=None, halo_bias=HBF,
                               mass_def=M200)

    # Wrong hbf
    with pytest.raises(TypeError):
        ccl.halos.HMCalculator(mass_function=HMF, halo_bias=None,
                               mass_def=M200)

    # Wrong MassDef
    with pytest.raises(TypeError):
        ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                               mass_def=None)

    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200)

    # Wrong profile
    with pytest.raises(TypeError):
        ccl.halos.halomod_mean_profile_1pt(COSMO, hmc, KK, AA, None,
                                           normprof=False)
    with pytest.raises(TypeError):
        ccl.halos.halomod_bias_1pt(COSMO, hmc, KK, AA, None,
                                   normprof=False)
    with pytest.raises(TypeError):
        ccl.halos.halomod_power_spectrum(COSMO, hmc, KK, AA, None,
                                         normprof=False)

    # Wrong prof2
    with pytest.raises(TypeError):
        ccl.halos.halomod_power_spectrum(COSMO, hmc, KK, AA, P1,
                                         prof2=KK, normprof=False)

    # Wrong prof_2pt
    with pytest.raises(TypeError):
        ccl.halos.halomod_power_spectrum(COSMO, hmc, KK, AA, P1,
                                         prof_2pt=KK, normprof=False)

    # Wrong pk2d
    with pytest.raises(TypeError):
        ccl.halos.halomod_power_spectrum(COSMO, hmc, KK, AA, P1,
                                         p_of_k_a=KK, normprof=False)

    def func():
        pass

    # Wrong 1h/2h smoothing
    with pytest.raises(TypeError):
        ccl.halos.halomod_power_spectrum(COSMO, hmc, KK, AA, P1,
                                         normprof=False,
                                         smooth_transition=True)
    with pytest.raises(ValueError):
        ccl.halos.halomod_power_spectrum(COSMO, hmc, KK, AA, P1,
                                         normprof=False,
                                         smooth_transition=func, get_1h=False)

    # Wrong 1h damping
    with pytest.raises(TypeError):
        ccl.halos.halomod_power_spectrum(COSMO, hmc, KK, AA, P1,
                                         normprof=False,
                                         suppress_1h=True)

    with pytest.raises(ValueError):
        ccl.halos.halomod_power_spectrum(COSMO, hmc, KK, AA, P1,
                                         normprof=False,
                                         suppress_1h=func, get_1h=False)
