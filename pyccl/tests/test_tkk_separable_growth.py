import numpy as np
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


def test_Tk3D_separable_growth():
    # tests the accuracy of the separable growth approximation
    # for trispectrum calculations, both for direct calls to
    # the functions and through the assmebly of Tk3D objects.
    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200)
    k_arr = KK
    a_arr = np.array([0.1, 0.4, 0.7, 1.0])

    tkk_arr = ccl.halos.halomod_trispectrum_2h_22(COSMO, hmc, k_arr, a_arr,
                                                  P1, prof2=P2, prof3=P3,
                                                  prof4=P4, prof13_2pt=PKC,
                                                  prof14_2pt=PKC,
                                                  prof24_2pt=PKC,
                                                  prof32_2pt=PKC,
                                                  p_of_k_a=None)

    tkk_arr_2 = ccl.halos.halomod_trispectrum_2h_22(COSMO, hmc, k_arr, a_arr,
                                                    P1, prof2=P2, prof3=P3,
                                                    prof4=P4, prof13_2pt=PKC,
                                                    prof14_2pt=PKC,
                                                    prof24_2pt=PKC,
                                                    prof32_2pt=PKC,
                                                    p_of_k_a=None,
                                                    separable_growth=True)

    tk3d = ccl.halos.halomod_Tk3D_2h(COSMO, hmc,
                                     P1, prof2=P2,
                                     prof3=P3, prof4=P4,
                                     prof13_2pt=PKC,
                                     prof14_2pt=PKC,
                                     prof24_2pt=PKC,
                                     prof32_2pt=PKC,
                                     p_of_k_a=None,
                                     a_arr=a_arr,
                                     lk_arr=np.log(k_arr),
                                     use_log=True)

    tk3d_2 = ccl.halos.halomod_Tk3D_2h(COSMO, hmc,
                                       P1, prof2=P2,
                                       prof3=P3, prof4=P4,
                                       prof13_2pt=PKC,
                                       prof14_2pt=PKC,
                                       prof24_2pt=PKC,
                                       prof32_2pt=PKC,
                                       p_of_k_a=None,
                                       a_arr=a_arr,
                                       lk_arr=np.log(k_arr),
                                       use_log=True,
                                       separable_growth=True)
    tkk_arr_3 = np.array([tk3d(k_arr, a) for a in a_arr])
    tkk_arr_4 = np.array([tk3d_2(k_arr, a) for a in a_arr])
    assert np.all(np.fabs((tkk_arr / tkk_arr_2 - 1)).flatten()
                  < 1E-4)
    assert np.all(np.fabs((tkk_arr_3 / tkk_arr_4 - 1)).flatten()
                  < 1E-4)

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

    tkk_arr_2 = ccl.halos.halomod_trispectrum_3h(COSMO, hmc, k_arr, a_arr,
                                                 prof=P1,
                                                 prof2=P2,
                                                 prof3=P3,
                                                 prof4=P4,
                                                 prof13_2pt=PKC,
                                                 prof14_2pt=PKC,
                                                 prof24_2pt=PKC,
                                                 prof32_2pt=PKC,
                                                 p_of_k_a=None,
                                                 separable_growth=True)

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

    tk3d_2 = ccl.halos.halomod_Tk3D_3h(COSMO, hmc,
                                       P1, prof2=P2,
                                       prof3=P3, prof4=P4,
                                       prof13_2pt=PKC,
                                       prof14_2pt=PKC,
                                       prof24_2pt=PKC,
                                       prof32_2pt=PKC,
                                       p_of_k_a=None,
                                       lk_arr=np.log(k_arr),
                                       a_arr=a_arr,
                                       use_log=True,
                                       separable_growth=True)

    tkk_arr_3 = np.array([tk3d(k_arr, a) for a in a_arr])
    tkk_arr_4 = np.array([tk3d_2(k_arr, a) for a in a_arr])
    assert np.all(np.fabs((tkk_arr / tkk_arr_2 - 1)).flatten()
                  < 1E-4)
    assert np.all(np.fabs((tkk_arr_3 / tkk_arr_4 - 1)).flatten()
                  < 1E-4)

    tkk_arr = ccl.halos.halomod_trispectrum_4h(COSMO, hmc, k_arr, a_arr,
                                               prof=P1,
                                               prof2=P2,
                                               prof3=P3,
                                               prof4=P4,
                                               p_of_k_a=None)

    tkk_arr_2 = ccl.halos.halomod_trispectrum_4h(COSMO, hmc, k_arr, a_arr,
                                                 prof=P1,
                                                 prof2=P2,
                                                 prof3=P3,
                                                 prof4=P4,
                                                 p_of_k_a=None,
                                                 separable_growth=True)

    tk3d = ccl.halos.halomod_Tk3D_4h(COSMO, hmc,
                                     P1, prof2=P2,
                                     prof3=P3, prof4=P4,
                                     p_of_k_a=None,
                                     lk_arr=np.log(k_arr),
                                     a_arr=a_arr,
                                     use_log=True)

    tk3d_2 = ccl.halos.halomod_Tk3D_4h(COSMO, hmc,
                                       P1, prof2=P2,
                                       prof3=P3, prof4=P4,
                                       p_of_k_a=None,
                                       lk_arr=np.log(k_arr),
                                       a_arr=a_arr,
                                       use_log=True,
                                       separable_growth=True)
    tkk_arr_3 = np.array([tk3d(k_arr, a) for a in a_arr])
    tkk_arr_4 = np.array([tk3d_2(k_arr, a) for a in a_arr])
    assert np.all(np.fabs((tkk_arr / tkk_arr_2 - 1)).flatten()
                  < 1E-4)
    assert np.all(np.fabs((tkk_arr_3 / tkk_arr_4 - 1)).flatten()
                  < 1E-4)

    tk3d = ccl.halos.halomod_Tk3D_cNG(COSMO, hmc,
                                      P1, prof2=P2,
                                      prof3=P3, prof4=P4,
                                      prof12_2pt=PKC,
                                      prof13_2pt=PKC,
                                      prof14_2pt=PKC,
                                      prof24_2pt=PKC,
                                      prof32_2pt=PKC,
                                      prof34_2pt=PKC,
                                      p_of_k_a=None,
                                      lk_arr=np.log(k_arr),
                                      a_arr=a_arr,
                                      use_log=True)

    tk3d_2 = ccl.halos.halomod_Tk3D_cNG(COSMO, hmc,
                                        P1, prof2=P2,
                                        prof3=P3, prof4=P4,
                                        prof12_2pt=PKC,
                                        prof13_2pt=PKC,
                                        prof14_2pt=PKC,
                                        prof24_2pt=PKC,
                                        prof32_2pt=PKC,
                                        prof34_2pt=PKC,
                                        p_of_k_a=None,
                                        lk_arr=np.log(k_arr),
                                        a_arr=a_arr,
                                        use_log=True,
                                        separable_growth=True)

    tkk_arr = np.array([tk3d(k_arr, a) for a in a_arr])
    tkk_arr_2 = np.array([tk3d_2(k_arr, a) for a in a_arr])
    assert np.all(np.fabs((tkk_arr / tkk_arr_2 - 1)).flatten()
                  < 1E-4)
