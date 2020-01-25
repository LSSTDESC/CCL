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
PKC = ccl.halos.ProfileCovar()
KK = np.geomspace(1E-3, 10, 32)
MM = np.geomspace(1E11, 1E15, 16)
AA = 1.0
PK2D = ccl.Pk2D(cosmo=COSMO, pkfunc=lambda k, a: a / k)


def test_profcovar_smoke():
    uk_NFW = P1.fourier(COSMO, KK, MM, AA,
                        mass_def=M200)
    uk_EIN = P2.fourier(COSMO, KK, MM, AA,
                        mass_def=M200)
    # Variance
    cv_NN = PKC.fourier_covar(P1, COSMO, KK, MM, AA,
                              mass_def=M200)
    assert np.all(np.fabs((cv_NN - uk_NFW**2)) < 1E-10)

    # Covariance
    cv_NE = PKC.fourier_covar(P1, COSMO, KK, MM, AA,
                              prof_2=P2, mass_def=M200)
    assert np.all(np.fabs((cv_NE - uk_NFW * uk_EIN)) < 1E-10)


def test_profcovar_errors():
    # Wrong first profile
    with pytest.raises(TypeError):
        PKC.fourier_covar(None, COSMO, KK, MM, AA,
                          prof_2=None, mass_def=M200)

    # Wrong second profile
    with pytest.raises(TypeError):
        PKC.fourier_covar(P1, COSMO, KK, MM, AA,
                          prof_2=M200, mass_def=M200)


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


@pytest.mark.parametrize('norm', [True, False])
def test_pkhm_mean_profile_smoke(norm):
    hmc = ccl.halos.HMCalculator(COSMO, nl10M=2)

    def f(k, a):
        return hmc.mean_profile(COSMO, k, a, HMF,
                                P1, normprof=norm,
                                mdef=M200)
    smoke_assert_pkhm_real(f)


@pytest.mark.parametrize('norm', [True, False])
def test_pkhm_bias_smoke(norm):
    hmc = ccl.halos.HMCalculator(COSMO, nl10M=2)

    def f(k, a):
        return hmc.bias(COSMO, k, a, HMF, HBF, P1,
                        normprof=norm, mdef=M200)
    smoke_assert_pkhm_real(f)


@pytest.mark.parametrize('pars',
                         [{'cv': None, 'norm': True,
                           'pk': 'linear', 'h1': True,
                           'h2': True, 'p2': None},
                          {'cv': PKC, 'norm': True,
                           'pk': 'linear', 'h1': True,
                           'h2': True, 'p2': None},
                          {'cv': None, 'norm': False,
                           'pk': 'linear', 'h1': True,
                           'h2': True, 'p2': None},
                          {'cv': None, 'norm': True,
                           'pk': 'nonlinear', 'h1': True,
                           'h2': True, 'p2': None},
                          {'cv': None, 'norm': True,
                           'pk': PK2D, 'h1': True,
                           'h2': True, 'p2': None},
                          {'cv': None, 'norm': True,
                           'pk': None, 'h1': True,
                           'h2': True, 'p2': None},
                          {'cv': None, 'norm': True,
                           'pk': 'linear', 'h1': False,
                           'h2': True, 'p2': None},
                          {'cv': None, 'norm': True,
                           'pk': 'linear', 'h1': True,
                           'h2': False, 'p2': None},
                          {'cv': None, 'norm': True,
                           'pk': 'linear', 'h1': False,
                           'h2': False, 'p2': None},
                          {'cv': None, 'norm': True,
                           'pk': 'linear', 'h1': True,
                           'h2': True, 'p2': P2}])
def test_pkhm_pk_smoke(pars):
    hmc = ccl.halos.HMCalculator(COSMO, nl10M=2)

    def f(k, a):
        return hmc.pk(COSMO, k, a, HMF, HBF, P1,
                      covprof=pars['cv'],
                      normprof_1=pars['norm'],
                      normprof_2=pars['norm'],
                      p_of_k_a=pars['pk'],
                      prof_2=pars['p2'],
                      mdef=M200,
                      get_1h=pars['h1'],
                      get_2h=pars['h2'])
    smoke_assert_pkhm_real(f)


def test_pkhm_pk2d():
    hmc = ccl.halos.HMCalculator(COSMO)
    k_arr = KK
    a_arr = np.linspace(0.1, 1, 10)
    pk_arr = hmc.pk(COSMO, k_arr, a_arr,
                    HMF, HBF, P1, mdef=M200,
                    normprof_1=True,
                    normprof_2=True)

    # Input sampling
    pk2d = hmc.get_Pk2D(COSMO, HMF, HBF, P1,
                        mdef=M200, lk_arr=np.log(k_arr),
                        a_arr=a_arr, normprof_1=True)
    pk_arr_2 = np.array([pk2d.eval(k_arr, a, COSMO)
                         for a in a_arr])
    assert np.all(np.fabs((pk_arr / pk_arr_2 - 1)).flatten()
                  < 1E-10)

    # Standard sampling
    pk2d = hmc.get_Pk2D(COSMO, HMF, HBF, P1,
                        mdef=M200, normprof_1=True)
    pk_arr_2 = np.array([pk2d.eval(k_arr, a, COSMO)
                         for a in a_arr])
    assert np.all(np.fabs((pk_arr / pk_arr_2 - 1)).flatten()
                  < 1E-10)


def test_pkhm_errors():
    # Wrong integration
    with pytest.raises(NotImplementedError):
        ccl.halos.HMCalculator(COSMO,
                               integration_method_M='Sampson')

    hmc = ccl.halos.HMCalculator(COSMO)

    # Wrong hmf
    with pytest.raises(TypeError):
        hmc.pk(COSMO, KK, AA, None, HBF, P1)

    # Wrong hbf
    with pytest.raises(TypeError):
        hmc.pk(COSMO, KK, AA, HMF, None, P1)

    # Wrong profile
    with pytest.raises(TypeError):
        hmc.pk(COSMO, KK, AA, HMF, HBF, None)

    # Wrong prof2
    with pytest.raises(TypeError):
        hmc.pk(COSMO, KK, AA, HMF, HBF, P1,
               prof_2=KK)

    # Wrong covprof
    with pytest.raises(TypeError):
        hmc.pk(COSMO, KK, AA, HMF, HBF, P1,
               covprof=KK)

    # Wrong pk2d
    with pytest.raises(TypeError):
        hmc.pk(COSMO, KK, AA, HMF, HBF, P1,
               p_of_k_a=KK)
