import os
import numpy as np
import pyccl as ccl
import pytest

TOLERANCE = 0.05

# Load response data
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "takahashi+2018")
RESPONSE_FILE = os.path.join(DATA_DIR, "pk3d_response_logkbin0.20_cic_mmin12_sum100_ng2048_010.txt")

RESPONSE_DATA = np.genfromtxt(RESPONSE_FILE).T
RESPONSE_WAVENUMBER = RESPONSE_DATA[0]
RESPONSE_VALUES = RESPONSE_DATA[1]
RESPONSE_K = 2 * np.pi / RESPONSE_WAVENUMBER
resp_k = RESPONSE_K
resp_data = RESPONSE_VALUES

@pytest.mark.parametrize(
    'cosmo,deltah',
    [(ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.7,
                          A_s=2.0e-9, n_s=0.96,
                          transfer_function='boltzmann_camb',
                          matter_power_spectrum='camb'), 0.02)])
def test_pk_response(cosmo, deltah):
    lk_arr = RESPONSE_WAVENUMBER
    a_arr = np.linspace(0.5, 1, 10)

    # Set up cosmology and 3D power spectrum
    #cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)
    #deltah = 0.02
    # lk_arr = np.logspace(-2, 1, 100)
    # a_arr = np.linspace(0.5, 1, 10)
    tk3dssc = ccl.Tk3D_SSC_Terasawa22(cosmo, deltah=deltah, lk_arr=lk_arr, a_arr=a_arr)

    # Calculate power spectrum response
    k_use = np.exp(lk_arr)
    pk = cosmo.get_nonlinear_matter_power_spectrum(k_use, a_arr[-1])
    dlog_pk_dlogk = np.gradient(np.log(pk), np.log(k_use))
    dP_dlogk = (1 + (26/21) * tk3d.T_growth(k_use) - (1/3) * dlog_pk_dlogk) * pk

    # # Load response data
    # DATA_DIR = 'data'
    # RESPONSE_FILE = os.path.join(DATA_DIR, 'takahashi+2018/pk3d_response_logkbin0.20_cic_mmin12_sum100_ng2048_010.txt')
    # RESPONSE_DATA = np.genfromtxt(RESPONSE_FILE).T
    # RESPONSE_WAVENUMBER = 10 ** RESPONSE_DATA[0]
    # RESPONSE_VALUES = RESPONSE_DATA[1]
    # RESPONSE_K = 2 * np.pi / RESPONSE_WAVENUMBER
    # resp_k = RESPONSE_K
    # resp_data = RESPONSE_VALUES

    # Compare with response data within 5% tolerance
    rel_diff = np.abs((dP_dlogk - resp_data) / resp_data)
    # tolerance = 0.05
    assert np.all(rel_diff < TOLERANCE)