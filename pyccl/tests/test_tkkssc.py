import numpy as np
import pyccl as ccl


COSMO = ccl.CosmologyVanillaLCDM(transfer_function='bbks',
                                 matter_power_spectrum='linear')
COSMO.compute_nonlin_power()
M200 = ccl.halos.MassDef200m()
HMF = ccl.halos.MassFuncTinker10(COSMO, M200)
HBF = ccl.halos.HaloBiasTinker10(COSMO, M200)
CON = ccl.halos.ConcentrationDuffy08(M200)
P1 = ccl.halos.HaloProfileNFW(CON, fourier_analytic=True)
KK = np.geomspace(1E-3, 10, 32)


def test_tkkssc_linear_bias():
    hmc = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                                 mass_def=M200, nlog10M=2)
    k_arr, a_arr = KK, np.linspace(0.3, 1.0, 4)
    bias1, bias2, bias3, bias4 = 2, 3, 4, 5

    # Test that if we remove the biases 12 will be the same as 34.
    tkkl = ccl.halos.halomod_Tk3D_SSC_linear_bias(
        COSMO, hmc, prof=P1,
        bias1=bias1, bias2=bias2, bias3=bias3, bias4=bias4,
        is_number_counts1=False, is_number_counts2=False,
        is_number_counts3=False, is_number_counts4=False,
        lk_arr=np.log(k_arr), a_arr=a_arr)

    _, _, _, (tkkl_12, tkkl_34) = tkkl.get_spline_arrays()
    tkkl_12 /= (bias1 * bias2)
    tkkl_34 /= (bias3 * bias4)
    assert np.allclose(tkkl_12, tkkl_34, atol=0, rtol=1e-12)

    # Test with the full T(k1,k2,a) for an NFW profile with bias ~1.
    tkk = ccl.halos.halomod_Tk3D_SSC(
        COSMO, hmc, prof1=P1, lk_arr=np.log(k_arr), a_arr=a_arr)
    _, _, _, (tkk_12, tkk_34) = tkk.get_spline_arrays()
    assert np.allclose(tkkl_12, tkk_12, atol=0, rtol=5e-3)
    assert np.allclose(tkkl_34, tkk_34, atol=0, rtol=5e-3)
