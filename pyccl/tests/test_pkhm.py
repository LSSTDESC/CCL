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

@pytest.mark.parametrize('func', ['mean_profile',
                                  'bias', 'pk'])
def test_pkhm_smoke(func):
    hmc = ccl.halos.HMCalculator()

    if func == 'mean_profile':
        def f(k, a):
            return hmc.mean_profile(COSMO, k, a,
                                    HMF, P1, normprof=True,
                                    mdef=M200)
    elif func == 'bias':
        def f(k, a):
            return hmc.bias(COSMO, k, a, HMF, HBF, P1,
                            normprof=True, mdef=M200)
    elif func == 'pk':
        def f(k, a):
            return hmc.pk(COSMO, k, a, HMF, HBF, P1,
                          covprof=PKC, prof_2=P2,
                          p_of_k_a='linear', normprof=True,
                          mdef=M200, get_1h=True, get_2h=True)
    smoke_assert_pkhm_real(f)
