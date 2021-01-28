import numpy as np
import pytest
import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
M200 = ccl.halos.MassDef200m()
HMF = ccl.halos.MassFuncTinker10(COSMO, mass_def=M200)
HBF = ccl.halos.HaloBiasTinker10(COSMO, mass_def=M200)
P1 = ccl.halos.HaloProfileNFW(ccl.halos.ConcentrationDuffy08(M200),
                              fourier_analytic=True)
P2 = P1
PKC = ccl.halos.Profile2pt()
KK = np.geomspace(1E-3, 10, 32)
MM = np.geomspace(1E11, 1E15, 16)
AA = 1.0
PK2D = ccl.Pk2D(cosmo=COSMO, pkfunc=lambda k, a: a / k)


def test_prof2pt_smoke():
    uk_NFW = P1.fourier(COSMO, KK, MM, AA,
                        mass_def=M200)
    uk_EIN = P2.fourier(COSMO, KK, MM, AA,
                        mass_def=M200)
    # Variance
    cv_NN = PKC.fourier_2pt(P1, COSMO, KK, MM, AA,
                            mass_def=M200)
    assert np.all(np.fabs((cv_NN - uk_NFW**2)) < 1E-10)

    # 2-point
    cv_NE = PKC.fourier_2pt(P1, COSMO, KK, MM, AA,
                            prof2=P2, mass_def=M200)
    assert np.all(np.fabs((cv_NE - uk_NFW * uk_EIN)) < 1E-10)


def test_prof2pt_errors():
    # Wrong first profile
    with pytest.raises(TypeError):
        PKC.fourier_2pt(None, COSMO, KK, MM, AA,
                        prof2=None, mass_def=M200)

    # Wrong second profile
    with pytest.raises(TypeError):
        PKC.fourier_2pt(P1, COSMO, KK, MM, AA,
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
    hmc = ccl.halos.HMCalculator(COSMO, HMF, HBF, mass_def=M200,
                                 nlog10M=2)

    def f(k, a):
        return ccl.halos.halomod_mean_profile_1pt(COSMO, hmc, k, a,
                                                  P1, normprof=norm)
    smoke_assert_pkhm_real(f)


@pytest.mark.parametrize('norm', [True, False])
def test_pkhm_bias_smoke(norm):
    hmc = ccl.halos.HMCalculator(COSMO, HMF, HBF, mass_def=M200,
                                 nlog10M=2)

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
                           'p2': P2}])
def test_pkhm_pk_smoke(pars):
    hmc = ccl.halos.HMCalculator(COSMO, HMF, HBF, mass_def=M200,
                                 nlog10M=2)

    def f(k, a):
        return ccl.halos.halomod_power_spectrum(COSMO, hmc, k, a, P1,
                                                prof_2pt=pars['cv'],
                                                normprof1=pars['norm'],
                                                normprof2=pars['norm'],
                                                p_of_k_a=pars['pk'],
                                                prof2=pars['p2'],
                                                get_1h=pars['h1'],
                                                get_2h=pars['h2'])
    smoke_assert_pkhm_real(f)


def test_pkhm_pk2d():
    hmc = ccl.halos.HMCalculator(COSMO, HMF, HBF, mass_def=M200)
    k_arr = KK
    a_arr = np.linspace(0.3, 1, 10)
    pk_arr = ccl.halos.halomod_power_spectrum(COSMO, hmc, k_arr, a_arr,
                                              P1, normprof1=True,
                                              normprof2=True)

    # Input sampling
    pk2d = ccl.halos.halomod_Pk2D(COSMO, hmc, P1,
                                  lk_arr=np.log(k_arr),
                                  a_arr=a_arr, normprof1=True)
    pk_arr_2 = np.array([pk2d.eval(k_arr, a, COSMO)
                         for a in a_arr])
    assert np.all(np.fabs((pk_arr / pk_arr_2 - 1)).flatten()
                  < 1E-4)

    # Standard sampling
    pk2d = ccl.halos.halomod_Pk2D(COSMO, hmc, P1,
                                  normprof1=True)
    pk_arr_2 = np.array([pk2d.eval(k_arr, a, COSMO)
                         for a in a_arr])
    assert np.all(np.fabs((pk_arr / pk_arr_2 - 1)).flatten()
                  < 1E-4)


def test_pkhm_errors():
    # Wrong integration
    with pytest.raises(NotImplementedError):
        ccl.halos.HMCalculator(COSMO, HMF, HBF, mass_def=M200,
                               integration_method_M='Sampson')

    # Wrong hmf
    with pytest.raises(TypeError):
        ccl.halos.HMCalculator(COSMO, None, HBF, mass_def=M200)

    # Wrong hbf
    with pytest.raises(TypeError):
        ccl.halos.HMCalculator(COSMO, HMF, None, mass_def=M200)

    hmc = ccl.halos.HMCalculator(COSMO, HMF, HBF, mass_def=M200)

    # Wrong profile
    with pytest.raises(TypeError):
        ccl.halos.halomod_mean_profile_1pt(COSMO, hmc, KK, AA, None)
    with pytest.raises(TypeError):
        ccl.halos.halomod_bias_1pt(COSMO, hmc, KK, AA, None)
    with pytest.raises(TypeError):
        ccl.halos.halomod_power_spectrum(COSMO, hmc, KK, AA, None)

    # Wrong prof2
    with pytest.raises(TypeError):
        ccl.halos.halomod_power_spectrum(COSMO, hmc, KK, AA, P1,
                                         prof2=KK)

    # Wrong prof_2pt
    with pytest.raises(TypeError):
        ccl.halos.halomod_power_spectrum(COSMO, hmc, KK, AA, P1,
                                         prof_2pt=KK)

    # Wrong pk2d
    with pytest.raises(TypeError):
        ccl.halos.halomod_power_spectrum(COSMO, hmc, KK, AA, P1,
                                         p_of_k_a=KK)
