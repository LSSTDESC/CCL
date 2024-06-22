import numpy as np
import os
from pyccl.pkresponse import Pmm_resp, darkemu_Pgm_resp, darkemu_Pgg_resp
import pyccl as ccl

data_directory_path = os.path.expanduser("benchmarks/data/SSC-Terasawa24/")

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


# Construct the full path for each data file (z=0)
k_data_path = os.path.join(data_directory_path, "k_h.npy")
k_data_mm_path = os.path.join(data_directory_path, "k_h_mm.npy")
Pmm_resp_data_path = os.path.join(data_directory_path, "Pmm_resp_z0.npy")
Pmm_resp_err_data_path = os.path.join(
    data_directory_path, "Pmm_resp_err_z0.npy"
)
Pgm_resp_data_path = os.path.join(data_directory_path, "Pgm_resp_z0.npy")
Pgm_resp_err_data_path = os.path.join(
    data_directory_path, "Pgm_resp_err_z0.npy"
)
Pgg_resp_data_path = os.path.join(data_directory_path, "Pgg_resp_z0.npy")
Pgg_resp_err_data_path = os.path.join(
    data_directory_path, "Pgg_resp_err_z0.npy"
)

# Load data
k_data = np.load(k_data_path)
k_data_mm = np.load(k_data_mm_path)
Pmm_resp_data = np.load(Pmm_resp_data_path) / h**3
Pmm_resp_err_data = np.load(Pmm_resp_err_data_path) / h**3
Pgm_resp_data = np.load(Pgm_resp_data_path) / h**3
Pgm_resp_err_data = np.load(Pgm_resp_err_data_path) / h**3
Pgg_resp_data = np.load(Pgg_resp_data_path) / h**3
Pgg_resp_err_data = np.load(Pgg_resp_err_data_path) / h**3


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
a_arr = np.array([1.0])
indx = (k_data > 1e-2) & (k_data < 4)
lk_arr = np.log(k_data[indx] * h)  # Using loaded k_data
indx_mm = (k_data_mm > 1e-2) & (k_data_mm < 4)
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
        Pmm_resp_data[indx_mm],
        generated_Pmm_resp,
        atol=6 * Pmm_resp_err_data[indx_mm],
    )


def test_pgm_resp():
    assert np.allclose(
        Pgm_resp_data[indx],
        generated_Pgm_resp,
        atol=2 * Pgm_resp_err_data[indx],
    )


def test_pgg_resp():
    assert np.allclose(
        Pgg_resp_data[indx],
        generated_Pgg_resp,
        atol=3 * Pgg_resp_err_data[indx],
    )
