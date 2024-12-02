import numpy as np
from pyccl.pkresponse import Pmm_resp, darkemu_Pgm_resp, darkemu_Pgg_resp
import pyccl as ccl

# Set cosmology
Om = 0.3156
ob = 0.02225
oc = 0.1198
OL = 0.6844
As = 2.2065e-9
ns = 0.9645
h = 0.6727

cosmo = ccl.Cosmology(
    Omega_c=oc / (h**2),
    Omega_b=ob / (h**2),
    h=h,
    n_s=ns,
    A_s=As,
    m_nu=0.06,
    transfer_function="boltzmann_camb",
    extra_parameters={"camb": {"halofit_version": "takahashi"}},
)


# Load data
pkresponse_data = np.load("benchmarks/data/pkresponse.npz")
k_data = pkresponse_data["k_h"]
k_data_mm = pkresponse_data["k_h_mm"]
Pmm_resp_data = pkresponse_data["Pmm_resp"] / h**3
Pmm_resp_err_data = pkresponse_data["Pmm_resp_err"] / h**3
Pgm_resp_data = pkresponse_data["Pgm_resp"] / h**3
Pgm_resp_err_data = pkresponse_data["Pgm_resp_err"] / h**3
Pgg_resp_data = pkresponse_data["Pgg_resp"] / h**3
Pgg_resp_err_data = pkresponse_data["Pgg_resp_err"]/ h**3


# HOD parameters
logMhmin = 13.94
logMh1 = 14.46
alpha = 1.192
kappa = 0.60
sigma_logM = 0.5
sigma_lM = sigma_logM * np.log(10)
logMh0 = logMhmin + np.log10(kappa)

logMmin = np.log10(10**logMhmin / h)
logM0 = np.log10(10**logMh0 / h)
logM1 = np.log10(10**logMh1 / h)

# Construct HOD
mass_def = ccl.halos.MassDef(200, "matter")
cm = ccl.halos.ConcentrationDuffy08(mass_def=mass_def)
prof_hod = ccl.halos.HaloProfileHOD(
    mass_def=mass_def,
    concentration=cm,
    log10Mmin_0=logMmin,
    siglnM_0=sigma_lM,
    log10M0_0=logM0,
    log10M1_0=logM1,
    alpha_0=alpha,
)

# Define input parameters for pkresponse functions
log10Mh_min = np.log10(2.6e12)
log10Mh_max = 15.9
a_arr = np.array([1.0, 0.64516129, 0.49382716, 0.40387722])
indx = (k_data > 1e-2) & (k_data < 3)
lk_arr = np.log(k_data[indx] * h)  # Using loaded k_data
indx_mm = (k_data_mm > 1e-2) & (k_data_mm < 3)
lk_arr_mm = np.log(k_data_mm[indx_mm] * h)  # Using loaded k_data

# Generate power spectrum responses using pkresponse.py functions
use_log = False

generated_Pmm_resp = Pmm_resp(
    cosmo, deltah=0.02, lk_arr=lk_arr_mm, a_arr=a_arr, use_log=use_log
)

generated_Pgm_resp = darkemu_Pgm_resp(
    cosmo,
    prof_hod,
    deltah=0.02,
    log10Mh_min=log10Mh_min,
    log10Mh_max=log10Mh_max,
    lk_arr=lk_arr,
    a_arr=a_arr,
    use_log=use_log,
)

generated_Pgg_resp = darkemu_Pgg_resp(
    cosmo,
    prof_hod,
    deltalnAs=0.03,
    log10Mh_min=log10Mh_min,
    log10Mh_max=log10Mh_max,
    lk_arr=lk_arr,
    a_arr=a_arr,
    use_log=use_log,
)


# Compare the generated responses with simulation data
def test_pmm_resp():
    assert np.allclose(
        Pmm_resp_data[0, indx_mm],
        generated_Pmm_resp[0],
        atol=10 * Pmm_resp_err_data[0, indx_mm],
    )

def test_pmm_resp2():
    assert np.allclose(
        Pmm_resp_data[1:, indx_mm],
        generated_Pmm_resp[1:],
        atol=60 * Pmm_resp_err_data[1:, indx_mm],
    )


def test_pgm_resp():
    assert np.allclose(
        Pgm_resp_data[:2, indx],
        generated_Pgm_resp[:2],
        atol=15 * Pgm_resp_err_data[:2, indx],
    )

def test_pgm_resp2():
    assert np.allclose(
        Pgm_resp_data[2:, indx],
        generated_Pgm_resp[2:],
        atol=50 * Pgm_resp_err_data[2:, indx],
    )


def test_pgg_resp():
    assert np.allclose(
        Pgg_resp_data[:2, indx],
        generated_Pgg_resp[:2],
        atol=4 * Pgg_resp_err_data[:2, indx],
    )

def test_pgg_resp2():
    assert np.allclose(
        Pgg_resp_data[2:, indx],
        generated_Pgg_resp[2:],
        atol=15 * Pgg_resp_err_data[2:, indx],
    )