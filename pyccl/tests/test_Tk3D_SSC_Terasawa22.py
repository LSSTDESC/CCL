import pyccl as ccl
import numpy as np


def test_tk3d_ssc_terasawa22():
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.7,
                          A_s=2.0e-9, n_s=0.96,
                          transfer_function='boltzmann_camb',
                          matter_power_spectrum='camb')
    a_arr = np.linspace(0.5, 1.0, 10)
    lk_arr = np.logspace(-2, 1, 100)

    # Create Tk3D object
    tk3dssc = ccl.tk3d.Tk3D_SSC_Terasawa22(cosmo=cosmo, lk_arr=lk_arr,
                                           a_arr=a_arr, extrap_order_lok=1,
                                           extrap_order_hik=1,
                                           use_log=False, deltah=0.02)

    # Evaluate trispectrum
    k = 0.1
    a = 0.7
    trisp = tk3dssc.eval(k, a)

    # Check result
    assert np.all(np.isfinite(trisp))
