import numpy as np
import pytest
import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
M200 = ccl.halos.MassDef200m
HMF = ccl.halos.MassFuncTinker10(mass_def=M200)
HBF = ccl.halos.HaloBiasTinker10(mass_def=M200)
CON = ccl.halos.ConcentrationDuffy08(mass_def=M200)
P1 = ccl.halos.HaloProfileNFW(mass_def=M200, concentration=CON,
                              fourier_analytic=True)
P2 = ccl.halos.HaloProfileHOD(mass_def=M200, concentration=CON)
P3 = ccl.halos.HaloProfilePressureGNFW(mass_def=M200)
P4 = P1
Pneg = ccl.halos.HaloProfilePressureGNFW(mass_def=M200, P0=-1)
PKC = ccl.halos.Profile2pt()
PKCH = ccl.halos.Profile2ptHOD()
KK = np.geomspace(1E-3, 10, 32)
MM = np.geomspace(1E11, 1E15, 16)
AA = 1.0
PSP = COSMO.get_nonlin_power()


def smoke_assert_tkk3h_real(func):
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
                           'cv13': None, 'cv14': None, 'cv24': None, 'cv32':
                           None, 'p_of_k_a': None},
                          {'p1': P1, 'p2': None, 'p3': None, 'p4': None,
                           'cv13': None, 'cv14': None, 'cv24': None, 'cv32':
                           None, 'p_of_k_a': None},
                          {'p1': P1, 'p2': P2, 'p3': None, 'p4': None,
                           'cv13': None, 'cv14': None, 'cv24': None, 'cv32':
                           None, 'p_of_k_a': None},
                          {'p1': P1, 'p2': P2, 'p3': P3, 'p4': P4,
                           'cv13': None, 'cv14': None, 'cv24': None, 'cv32':
                           None, 'p_of_k_a': None},
                          {'p1': P1, 'p2': None, 'p3': None, 'p4': P4,
                           'cv13': None, 'cv14': None, 'cv24': None, 'cv32':
                           None, 'p_of_k_a': None},
                          {'p1': P1, 'p2': P2, 'p3': P3, 'p4': P2,
                           'cv13': None, 'cv14': None, 'cv24': PKCH, 'cv32':
                           None, 'p_of_k_a': None},
                          {'p1': P1, 'p2': P2, 'p3': P3, 'p4': P2,
                           'cv13': PKC, 'cv14': None, 'cv24': PKCH, 'cv32':
                           None, 'p_of_k_a': None},
                          {'p1': P1, 'p2': P2, 'p3': P3, 'p4': P4,
                           'cv13': PKC, 'cv14': None, 'cv24': PKC, 'cv32':
                           None, 'p_of_k_a': None},
                          {'p1': P1, 'p2': P2, 'p3': P2, 'p4': P4,
                           'cv13': PKC, 'cv14': PKC, 'cv24': PKC, 'cv32':
                           PKCH, 'p_of_k_a': None},
                          {'p1': P1, 'p2': P2, 'p3': P2, 'p4': P4,
                           'cv13': PKC, 'cv14': PKC, 'cv24': PKC, 'cv32':
                           PKCH, 'p_of_k_a': 'linear'},
                          {'p1': P1, 'p2': P2, 'p3': P2, 'p4': P4,
                           'cv13': PKC, 'cv14': PKC, 'cv24': PKC, 'cv32':
                           PKCH, 'p_of_k_a': 'nonlinear'},
                          {'p1': P1, 'p2': P2, 'p3': P2, 'p4': P4,
                           'cv13': PKC, 'cv14': PKC, 'cv24': PKC, 'cv32':
                           PKCH, 'p_of_k_a': PSP},
                          ])
def test_tkk3h_smoke(pars):
    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200, nM=2)

    def f(k, a):
        return ccl.halos.halomod_trispectrum_3h(COSMO, hmc, k, a,
                                                prof=pars['p1'],
                                                prof2=pars['p2'],
                                                prof3=pars['p3'],
                                                prof4=pars['p4'],
                                                prof13_2pt=pars['cv13'],
                                                prof14_2pt=pars['cv14'],
                                                prof24_2pt=pars['cv24'],
                                                prof32_2pt=pars['cv32'],
                                                p_of_k_a=pars['p_of_k_a'])
    smoke_assert_tkk3h_real(f)


def test_Tk3D_3h():
    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200)
    k_arr = KK
    a_arr = np.array([0.1, 0.4, 0.7, 1.0])
    tkk_arr = ccl.halos.halomod_trispectrum_3h(COSMO, hmc, k_arr, a_arr,
                                               prof=P1,
                                               prof2=P2,
                                               prof3=P3,
                                               prof4=P4,
                                               prof13_2pt=PKC,
                                               prof14_2pt=PKC,
                                               prof24_2pt=PKC,
                                               prof32_2pt=PKC,
                                               p_of_k_a=None)
    # Input sampling
    tk3d = ccl.halos.halomod_Tk3D_3h(COSMO, hmc,
                                     P1, prof2=P2,
                                     prof3=P3, prof4=P4,
                                     prof13_2pt=PKC,
                                     prof14_2pt=PKC,
                                     prof24_2pt=PKC,
                                     prof32_2pt=PKC,
                                     p_of_k_a=None,
                                     lk_arr=np.log(k_arr),
                                     a_arr=a_arr,
                                     use_log=True)
    tkk_arr_2 = np.array([tk3d(k_arr, a) for a in a_arr])
    assert np.all(np.fabs((tkk_arr / tkk_arr_2 - 1)).flatten()
                  < 1E-4)

    # Standard sampling
    tk3d = ccl.halos.halomod_Tk3D_3h(COSMO, hmc,
                                     P1, prof2=P2,
                                     prof3=P3, prof4=P4,
                                     prof13_2pt=PKC,
                                     prof14_2pt=PKC,
                                     prof24_2pt=PKC,
                                     prof32_2pt=PKC,
                                     p_of_k_a=None,
                                     lk_arr=np.log(k_arr),
                                     use_log=True)
    tkk_arr_2 = np.array([tk3d(k_arr, a) for a in a_arr])
    assert np.all(np.fabs((tkk_arr / tkk_arr_2 - 1)).flatten()
                  < 1E-4)

    # Negative profile in logspace
    with pytest.warns(ccl.CCLWarning):
        ccl.halos.halomod_Tk3D_3h(COSMO, hmc, P3, prof2=Pneg, prof3=P3,
                                  prof4=P3, lk_arr=np.log(k_arr), a_arr=a_arr,
                                  use_log=True)


@pytest.mark.parametrize('pars',
                         # Wrong first profile
                         [{'p1': None, 'p2': None, 'p3': None, 'p4': None,
                           'cv13': None, 'cv14': None, 'cv24': None, 'cv32':
                           None, 'p_of_k_a': None},
                          # Wrong other profiles
                          {'p1': P1, 'p2': PKC, 'p3': None, 'p4': None,
                           'cv13': None, 'cv14': None, 'cv24': None, 'cv32':
                           None, 'p_of_k_a': None},
                          {'p1': P1, 'p2': None, 'p3': PKC, 'p4': None,
                           'cv13': None, 'cv14': None, 'cv24': None, 'cv32':
                           None, 'p_of_k_a': None},
                          {'p1': P1, 'p2': None, 'p3': None, 'p4': PKC,
                           'cv13': None, 'cv14': None, 'cv24': None, 'cv32':
                           None, 'p_of_k_a': None},
                          {'p1': P1, 'p2': None, 'p3': None, 'p4': None,
                           'cv13': P2, 'cv14': None, 'cv24': None, 'cv32':
                           None, 'p_of_k_a': None},
                          # Wrong 2pts
                          {'p1': P1, 'p2': None, 'p3': None, 'p4': None,
                           'cv13': None, 'cv14': P2, 'cv24': None, 'cv32':
                           None, 'p_of_k_a': None},
                          {'p1': P1, 'p2': None, 'p3': None, 'p4': None,
                           'cv13': None, 'cv14': None, 'cv24': P2, 'cv32':
                           None, 'p_of_k_a': None},
                          {'p1': P1, 'p2': None, 'p3': None, 'p4': None,
                           'cv13': None, 'cv14': None, 'cv24': None, 'cv32':
                           P2, 'p_of_k_a': None},
                          # Wrong p_of_k_a
                          {'p1': P1, 'p2': None, 'p3': None, 'p4': None,
                           'cv13': None, 'cv14': None, 'cv24': None, 'cv32':
                           None, 'p_of_k_a': P2},
                          ])
def test_tkk3h_errors(pars):
    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200)
    k_arr = KK
    a_arr = np.array([0.1, 0.4, 0.7, 1.0])

    error = AttributeError if pars['p_of_k_a'] is None else TypeError
    with pytest.raises(error):
        ccl.halos.halomod_trispectrum_3h(COSMO, hmc, k_arr, a_arr,
                                         prof=pars['p1'], prof2=pars['p2'],
                                         prof3=pars['p3'], prof4=pars['p4'],
                                         prof13_2pt=pars['cv13'],
                                         prof14_2pt=pars['cv14'],
                                         prof24_2pt=pars['cv24'],
                                         prof32_2pt=pars['cv32'],
                                         p_of_k_a=pars['p_of_k_a'])
