import numpy as np
import pytest
import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
COSMO.compute_nonlin_power()
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
PKCH = ccl.halos.Profile2ptHOD()
KK = np.geomspace(1E-3, 10, 32)
MM = np.geomspace(1E11, 1E15, 16)
AA = 1.0


@pytest.mark.parametrize('pars',
                         [{'p1': P1, 'p2': None, 'cv12': None,
                           'p3': None, 'p4': None, 'cv34': None,
                           'norm': False, 'pk': None},
                          {'p1': P1, 'p2': None, 'cv12': None,
                           'p3': None, 'p4': None, 'cv34': None,
                           'norm': True, 'pk': None},
                          {'p1': P1, 'p2': P2, 'cv12': None,
                           'p3': None, 'p4': None, 'cv34': None,
                           'norm': False, 'pk': None},
                          {'p1': P1, 'p2': None, 'cv12': None,
                           'p3': P3, 'p4': None, 'cv34': None,
                           'norm': False, 'pk': None},
                          {'p1': P1, 'p2': None, 'cv12': None,
                           'p3': None, 'p4': P4, 'cv34': None,
                           'norm': False, 'pk': None},
                          {'p1': P1, 'p2': P2, 'cv12': None,
                           'p3': P3, 'p4': P4, 'cv34': None,
                           'norm': False, 'pk': None},
                          {'p1': P2, 'p2': P2, 'cv12': PKCH,
                           'p3': P2, 'p4': P2, 'cv34': None,
                           'norm': False, 'pk': None},
                          {'p1': P1, 'p2': P2, 'cv12': PKC,
                           'p3': P3, 'p4': P4, 'cv34': PKC,
                           'norm': True, 'pk': None},
                          {'p1': P1, 'p2': None, 'cv12': None,
                           'p3': None, 'p4': None, 'cv34': None,
                           'norm': False, 'pk': 'linear'},
                          {'p1': P1, 'p2': None, 'cv12': None,
                           'p3': None, 'p4': None, 'cv34': None,
                           'norm': False, 'pk': 'nonlinear'},
                          {'p1': P1, 'p2': None, 'cv12': None,
                           'p3': None, 'p4': None, 'cv34': None,
                           'norm': False, 'pk': COSMO.get_nonlin_power()}],)
def test_tkkssc_smoke(pars):
    hmc = ccl.halos.HMCalculator(COSMO, HMF, HBF, mass_def=M200,
                                 nlog10M=2)
    k_arr = KK
    a_arr = np.array([0.3, 0.5, 0.7, 1.0])

    tkk = ccl.halos.halomod_Tk3D_SSC(COSMO, hmc,
                                     prof1=pars['p1'],
                                     prof2=pars['p2'],
                                     prof12_2pt=pars['cv12'],
                                     prof3=pars['p3'],
                                     prof4=pars['p4'],
                                     prof34_2pt=pars['cv34'],
                                     normprof1=pars['norm'],
                                     normprof2=pars['norm'],
                                     normprof3=pars['norm'],
                                     normprof4=pars['norm'],
                                     p_of_k_a=pars['pk'],
                                     lk_arr=np.log(k_arr), a_arr=a_arr)
    tk = tkk.eval(0.1, 0.5)
    assert np.all(np.isfinite(tk))


def test_tkkssc_errors():
    from pyccl.pyutils import assert_warns

    hmc = ccl.halos.HMCalculator(COSMO, HMF, HBF, mass_def=M200)
    k_arr = KK
    a_arr = np.array([0.3, 0.5, 0.7, 1.0])

    # Wrong first profile
    with pytest.raises(TypeError):
        ccl.halos.halomod_Tk3D_SSC(COSMO, hmc, None)
    # Wrong other profiles
    for i in range(2, 4):
        kw = {'prof%d' % i: PKC}
        with pytest.raises(TypeError):
            ccl.halos.halomod_Tk3D_SSC(COSMO, hmc, P1, **kw)
    # Wrong 2pts
    with pytest.raises(TypeError):
        ccl.halos.halomod_Tk3D_SSC(COSMO, hmc, P1,
                                   prof12_2pt=P2)
    with pytest.raises(TypeError):
        ccl.halos.halomod_Tk3D_SSC(COSMO, hmc, P1,
                                   prof34_2pt=P2)

    # Negative profile in logspace
    assert_warns(ccl.CCLWarning, ccl.halos.halomod_Tk3D_1h,
                 COSMO, hmc, P3, prof2=Pneg,
                 lk_arr=np.log(k_arr), a_arr=a_arr,
                 use_log=True)
