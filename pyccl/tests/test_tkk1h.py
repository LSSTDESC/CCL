import numpy as np
import pytest
import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
M200 = ccl.halos.MassDef200m()
HMF = ccl.halos.MassFuncTinker10(mass_def=M200)
HBF = ccl.halos.HaloBiasTinker10(mass_def=M200)
CONC = ccl.halos.ConcentrationDuffy08(mass_def=M200)
P1 = ccl.halos.HaloProfileNFW(c_m_relation=CONC, fourier_analytic=True)
P2 = ccl.halos.HaloProfileHOD(c_m_relation=CONC)
P3 = ccl.halos.HaloProfilePressureGNFW()
P4 = P1
Pneg = ccl.halos.HaloProfilePressureGNFW(P0=-1)
PKC = ccl.halos.Profile2pt()
PKCH = ccl.halos.Profile2ptHOD()
KK = np.geomspace(1E-3, 10, 32)
MM = np.geomspace(1E11, 1E15, 16)
AA = 1.0


def smoke_assert_tkk1h_real(func):
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
                         [{'p1': P1, 'p2': None, 'cv12': None,
                           'p3': None, 'p4': None, 'cv34': None,
                           'norm': False},
                          {'p1': P1, 'p2': None, 'cv12': None,
                           'p3': None, 'p4': None, 'cv34': None,
                           'norm': True},
                          {'p1': P1, 'p2': P2, 'cv12': None,
                           'p3': None, 'p4': None, 'cv34': None,
                           'norm': False},
                          {'p1': P1, 'p2': None, 'cv12': None,
                           'p3': P3, 'p4': None, 'cv34': None,
                           'norm': False},
                          {'p1': P1, 'p2': None, 'cv12': None,
                           'p3': None, 'p4': P4, 'cv34': None,
                           'norm': False},
                          {'p1': P1, 'p2': P2, 'cv12': None,
                           'p3': P3, 'p4': P4, 'cv34': None,
                           'norm': False},
                          {'p1': P2, 'p2': P2, 'cv12': PKCH,
                           'p3': P2, 'p4': P2, 'cv34': None,
                           'norm': False},
                          {'p1': P1, 'p2': P2, 'cv12': PKC,
                           'p3': P3, 'p4': P4, 'cv34': PKC,
                           'norm': True}],)
def test_tkk1h_smoke(pars):
    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200, nlM=2)

    def f(k, a):
        return ccl.halos.halomod_trispectrum_1h(COSMO, hmc, k, a,
                                                prof=pars['p1'],
                                                prof2=pars['p2'],
                                                prof12_2pt=pars['cv12'],
                                                prof3=pars['p3'],
                                                prof4=pars['p4'],
                                                prof34_2pt=pars['cv34'],
                                                normprof=pars['norm'],
                                                normprof2=pars['norm'],
                                                normprof3=pars['norm'],
                                                normprof4=pars['norm'])
    smoke_assert_tkk1h_real(f)


def test_tkk1h_tk3d():
    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200)
    k_arr = KK
    a_arr = np.array([0.1, 0.4, 0.7, 1.0])
    tkk_arr = ccl.halos.halomod_trispectrum_1h(COSMO, hmc, k_arr, a_arr,
                                               P1, prof2=P2,
                                               prof12_2pt=PKC,
                                               prof3=P3, prof4=P4,
                                               prof34_2pt=PKC,
                                               normprof=True,
                                               normprof2=True,
                                               normprof3=True,
                                               normprof4=True)

    # Input sampling
    tk3d = ccl.halos.halomod_Tk3D_1h(COSMO, hmc,
                                     P1, prof2=P2,
                                     prof12_2pt=PKC,
                                     prof3=P3, prof4=P4,
                                     prof34_2pt=PKC,
                                     normprof=True,
                                     normprof2=True,
                                     normprof3=True,
                                     normprof4=True,
                                     lk_arr=np.log(k_arr),
                                     a_arr=a_arr,
                                     use_log=True)
    tkk_arr_2 = np.array([tk3d.eval(k_arr, a) for a in a_arr])
    assert np.allclose(tkk_arr, tkk_arr_2, rtol=1E-4)

    # Standard sampling
    with np.errstate(divide="ignore", invalid="ignore"):
        tk3d = ccl.halos.halomod_Tk3D_1h(COSMO, hmc,
                                         P1, prof2=P2,
                                         prof12_2pt=PKC,
                                         prof3=P3, prof4=P4,
                                         prof34_2pt=PKC,
                                         normprof=True,
                                         normprof2=True,
                                         normprof3=True,
                                         normprof4=True,
                                         lk_arr=np.log(k_arr),
                                         use_log=True)
    tkk_arr_2 = np.array([tk3d.eval(k_arr, a) for a in a_arr])
    assert np.allclose(tkk_arr, tkk_arr_2, rtol=1E-4)


def test_tkk1h_errors():
    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200)
    k_arr = KK
    a_arr = np.array([0.1, 0.4, 0.7, 1.0])

    # Wrong first profile
    with pytest.raises(TypeError):
        ccl.halos.halomod_trispectrum_1h(COSMO, hmc, k_arr, a_arr, None,
                                         normprof=False)
    # Wrong other profiles
    with pytest.raises(TypeError):
        ccl.halos.halomod_trispectrum_1h(COSMO, hmc, k_arr, a_arr,
                                         P1, prof2=PKC, normprof=False)
    with pytest.raises(TypeError):
        ccl.halos.halomod_trispectrum_1h(COSMO, hmc, k_arr, a_arr,
                                         P1, prof3=PKC, normprof=False)
    with pytest.raises(TypeError):
        ccl.halos.halomod_trispectrum_1h(COSMO, hmc, k_arr, a_arr,
                                         P1, prof4=PKC, normprof=False)
    # Wrong 2pts
    with pytest.raises(TypeError):
        ccl.halos.halomod_trispectrum_1h(COSMO, hmc, k_arr, a_arr,
                                         P1, prof12_2pt=P2, normprof=False)
    with pytest.raises(TypeError):
        ccl.halos.halomod_trispectrum_1h(COSMO, hmc, k_arr, a_arr,
                                         P1, prof34_2pt=P2, normprof=False)

    # Negative profile in logspace
    with pytest.warns(ccl.CCLWarning):
        ccl.halos.halomod_Tk3D_1h(
            COSMO, hmc, P3, prof2=Pneg, prof3=P3, prof4=P3, normprof=False,
            lk_arr=np.log(k_arr), a_arr=a_arr, use_log=True)
