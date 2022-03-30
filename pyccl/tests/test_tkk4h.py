import numpy as np
import pytest
import pyccl as ccl
from pyccl.pyutils import assert_warns


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
M200 = ccl.halos.MassDef200m()
HMF = ccl.halos.MassFuncTinker10(COSMO, mass_def=M200)
HBF = ccl.halos.HaloBiasTinker10(COSMO, mass_def=M200)
P1 = ccl.halos.HaloProfileNFW(ccl.halos.ConcentrationDuffy08(M200),
                              fourier_analytic=True)
P2 = ccl.halos.HaloProfileHOD(ccl.halos.ConcentrationDuffy08(M200))
P3 = ccl.halos.HaloProfilePressureGNFW()
P4 = P1
Pneg = ccl.halos.HaloProfilePressureGNFW(P0=-1)
PKC = ccl.halos.Profile2pt()
Prof3pt = ccl.halos.Profile3pt()
PKCH = ccl.halos.Profile2ptHOD()
KK = np.geomspace(1E-3, 10, 32)
MM = np.geomspace(1E11, 1E15, 16)
AA = 1.0
PSP = ccl.Pk2D.pk_from_model(COSMO, 'bbks')


def smoke_assert_tkk4h_real(func):
    sizes = [(0, 0),
             (2, 0),
             (0, 2),
             (2, 3),
             (1, 3),
             (3, 1)]
    shapes = [(),
              (2,),
              (2, 2,),
              (2, 3, 3),
              (1, 3, 3),
              (3, 1, 1)]
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


@pytest.mark.parametrize('pars',
                         [{'p1': P1, 'p2': None, 'p3': None, 'p4': None,
                           'norm': False, 'p_of_k_a': None},
                          {'p1': P1, 'p2': None, 'p3': None, 'p4': None,
                           'norm': True, 'p_of_k_a': None},
                          {'p1': P1, 'p2': P2, 'p3': None, 'p4': None,
                           'norm': True, 'p_of_k_a': None},
                          {'p1': P1, 'p2': P2, 'p3': P3, 'p4': P4,
                           'norm': True, 'p_of_k_a': None},
                          {'p1': P1, 'p2': None, 'p3': None, 'p4': P4,
                           'norm': True, 'p_of_k_a': None},
                          {'p1': P1, 'p2': P2, 'p3': P3, 'p4': P4,
                           'norm': True, 'p_of_k_a': None},
                          {'p1': P1, 'p2': P2, 'p3': P3, 'p4': P4,
                           'norm': True, 'p_of_k_a': 'linear'},
                          {'p1': P1, 'p2': P2, 'p3': P3, 'p4': P4,
                           'norm': True, 'p_of_k_a': 'nonlinear'},
                          {'p1': P1, 'p2': P2, 'p3': P3, 'p4': P4,
                           'norm': True, 'p_of_k_a': PSP},
                          ])
def test_tkk4h_smoke(pars):
    hmc = ccl.halos.HMCalculator(COSMO, HMF, HBF, mass_def=M200,
                                 nlog10M=2)

    def f(k, a):
        return ccl.halos.halomod_trispectrum_4h(COSMO, hmc, k, a,
                                                prof1=pars['p1'],
                                                prof2=pars['p2'],
                                                prof3=pars['p3'],
                                                prof4=pars['p4'],
                                                normprof1=pars['norm'],
                                                normprof2=pars['norm'],
                                                normprof3=pars['norm'],
                                                normprof4=pars['norm'],
                                                p_of_k_a=pars['p_of_k_a'])
    smoke_assert_tkk4h_real(f)


def test_Tk3D_4h():
    hmc = ccl.halos.HMCalculator(COSMO, HMF, HBF, mass_def=M200)
    k_arr = KK
    a_arr = np.array([0.1, 0.4, 0.7, 1.0])
    tkk_arr = ccl.halos.halomod_trispectrum_4h(COSMO, hmc, k_arr, a_arr,
                                               prof1=P1,
                                               prof2=P2,
                                               prof3=P3,
                                               prof4=P4,
                                               normprof1=True,
                                               normprof2=True,
                                               normprof3=True,
                                               normprof4=True,
                                               p_of_k_a=None)
    # Input sampling
    tk3d = ccl.halos.halomod_Tk3D_4h(COSMO, hmc,
                                     P1, prof2=P2,
                                     prof3=P3, prof4=P4,
                                     normprof1=True,
                                     normprof2=True,
                                     normprof3=True,
                                     normprof4=True,
                                     p_of_k_a=None,
                                     lk_arr=np.log(k_arr),
                                     a_arr=a_arr,
                                     use_log=True)
    tkk_arr_2 = np.array([tk3d.eval(k_arr, a) for a in a_arr])
    assert np.all(np.fabs((tkk_arr / tkk_arr_2 - 1)).flatten()
                  < 1E-4)

    # Standard sampling
    tk3d = ccl.halos.halomod_Tk3D_4h(COSMO, hmc,
                                     P1, prof2=P2,
                                     prof3=P3, prof4=P4,
                                     normprof1=True,
                                     normprof2=True,
                                     normprof3=True,
                                     normprof4=True,
                                     p_of_k_a=None,
                                     lk_arr=np.log(k_arr),
                                     use_log=True)
    tkk_arr_2 = np.array([tk3d.eval(k_arr, a) for a in a_arr])
    assert np.all(np.fabs((tkk_arr / tkk_arr_2 - 1)).flatten()
                  < 1E-4)

    # Negative profile in logspace
    assert_warns(ccl.CCLWarning, ccl.halos.halomod_Tk3D_2h,
                 COSMO, hmc, P3, prof2=Pneg,
                 lk_arr=np.log(k_arr), a_arr=a_arr,
                 use_log=True)


@pytest.mark.parametrize('pars',
                         # Wrong first profile
                         [{'p1': None, 'p2': None, 'p3': None, 'p4': None,
                           'p_of_k_a': None},
                          # Wrong other profiles
                          {'p1': P1, 'p2': PKC, 'p3': None, 'p4': None,
                           'p_of_k_a': None},
                          {'p1': P1, 'p2': None, 'p3': PKC, 'p4': None,
                           'p_of_k_a': None},
                          {'p1': P1, 'p2': None, 'p3': None, 'p4': PKC,
                           'p_of_k_a': None},
                          # Wron p_of_k_a
                          {'p1': P1, 'p2': None, 'p3': None, 'p4': None,
                           'p_of_k_a': P2},
                          ])
def test_tkk4h_errors(pars):

    hmc = ccl.halos.HMCalculator(COSMO, HMF, HBF, mass_def=M200)
    k_arr = KK
    a_arr = np.array([0.1, 0.4, 0.7, 1.0])

    with pytest.raises(TypeError):
        ccl.halos.halomod_trispectrum_4h(COSMO, hmc, k_arr, a_arr,
                                         prof1=pars['p1'], prof2=pars['p2'],
                                         prof3=pars['p3'], prof4=pars['p4'],
                                         p_of_k_a=pars['p_of_k_a'])
