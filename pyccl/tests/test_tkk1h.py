import numpy as np
import pytest
import pyccl as ccl


COSMO = ccl.CosmologyVanillaLCDM(transfer_function='bbks',
                                 matter_power_spectrum='linear')
M200 = ccl.halos.MassDef200m
HMF = ccl.halos.MassFuncTinker10(mass_def=M200)
HBF = ccl.halos.HaloBiasTinker10(mass_def=M200)
CON = ccl.halos.ConcentrationDuffy08(mass_def=M200)
P1 = ccl.halos.HaloProfileNFW(mass_def=M200, concentration=CON,
                              fourier_analytic=True)
P2 = ccl.halos.HaloProfileHOD(mass_def=M200, concentration=CON)
P3 = ccl.halos.HaloProfilePressureGNFW(mass_def=M200)
P4 = P1
PKC = ccl.halos.Profile2pt()
PKCH = ccl.halos.Profile2ptHOD()
KK = np.geomspace(1E-3, 10, 32)


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


@pytest.mark.parametrize(
    "p1,p2,cv12,p3,p4,cv34",
    [(P1, None, None, None, None, None),
     (P1, None, None, None, None, None),
     (P1, P2, None, None, None, None),
     (P1, None, None, P3, None, None),
     (P1, None, None, None, P4, None),
     (P1, P2, None, P3, P4, None),
     (P2, P2, PKCH, P2, P2, None),
     (P1, P2, PKC, P3, P4, PKC)])
def test_tkk1h_smoke(p1, p2, cv12, p3, p4, cv34):
    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200, nM=2)

    def f(k, a):
        return ccl.halos.halomod_trispectrum_1h(
            COSMO, hmc, k, a,
            prof=p1, prof2=p2, prof12_2pt=cv12,
            prof3=p3, prof4=p4, prof34_2pt=cv34)
    smoke_assert_tkk1h_real(f)


def test_tkk1h_tk3d():
    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200)
    k_arr = KK
    a_arr = np.array([0.1, 0.4, 0.7, 1.0])
    tkk_arr = ccl.halos.halomod_trispectrum_1h(
        COSMO, hmc, k_arr, a_arr,
        prof=P1, prof2=P2, prof12_2pt=PKC,
        prof3=P3, prof4=P4, prof34_2pt=PKC)

    # Input sampling
    tk3d = ccl.halos.halomod_Tk3D_1h(
        COSMO, hmc,
        prof=P1, prof2=P2, prof12_2pt=PKC,
        prof3=P3, prof4=P4, prof34_2pt=PKC,
        lk_arr=np.log(k_arr), a_arr=a_arr, use_log=True)

    tkk_arr_2 = tk3d(k_arr, a_arr)
    assert np.allclose(tkk_arr, tkk_arr_2, atol=0, rtol=1e-4)

    # Standard sampling
    tk3d = ccl.halos.halomod_Tk3D_1h(
        COSMO, hmc,
        prof=P1, prof2=P2, prof12_2pt=PKC,
        prof3=P3, prof4=P4, prof34_2pt=PKC,
        lk_arr=np.log(k_arr), use_log=True)

    tkk_arr_2 = tk3d(k_arr, a_arr)
    assert np.allclose(tkk_arr, tkk_arr_2, atol=0, rtol=1e-4)


def test_tkk1h_warns():
    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200)
    a_arr = np.array([0.1, 0.4, 0.7, 1.0])

    # Negative profile in logspace
    Pneg = ccl.halos.HaloProfilePressureGNFW(mass_def=M200, P0=-1)
    ccl.update_warning_verbosity('high')
    with pytest.warns(ccl.CCLWarning):
        ccl.halos.halomod_Tk3D_1h(
            COSMO, hmc, P3, prof2=Pneg, prof3=P3, prof4=P3,
            lk_arr=np.log(KK), a_arr=a_arr, use_log=True)
    ccl.update_warning_verbosity('low')
