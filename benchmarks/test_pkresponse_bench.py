import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the directory containing pkresponse.py to the Python path
sys.path.append(os.path.expanduser("../../pyccl"))

from pkresponse import darkemu_Pgm_resp, darkemu_Pgg_resp
import pyccl as ccl

data_directory_path = os.path.expanduser("../../benchmarks/data/SSC-Terasawa24/")

# Construct the full path for each data file (z=0)
k_data_path = os.path.join(data_directory_path, "k_h.npy")
Pgm_resp_data_path = os.path.join(data_directory_path, "Pgm_resp_z0.npy")
Pgm_resp_err_data_path = os.path.join(data_directory_path, "Pgm_resp_err_z0.npy")
Pgg_resp_data_path = os.path.join(data_directory_path, "Pgg_resp_z0.npy")
Pgg_resp_err_data_path = os.path.join(data_directory_path, "Pgg_resp_err_z0.npy")

# Load data
k_data = np.load(k_data_path)
Pgm_resp_data = np.load(Pgm_resp_data_path)
Pgm_resp_err_data = np.load(Pgm_resp_err_data_path)
Pgg_resp_data = np.load(Pgg_resp_data_path)
Pgg_resp_err_data = np.load(Pgg_resp_err_data_path)

# Set cosmology
cosmo_params = {
    "Omega_c": 0.27,
    "Omega_b": 0.045,
    "h": 0.67,
    "sigma8": 0.8,
    "n_s": 0.96,
}
cosmo = ccl.Cosmology(**cosmo_params)

# Placeholder, replace with actual halo model parameters if available
hmc = ccl.halos.MassFunc(ccl.halos.MassDef(200, 'critical'))
prof_hod = ccl.halos.HaloProfileHOD()

# Define input parameters for pkresponse functions
deltah = 0.02
log10Mh_min = 12.0
log10Mh_max = 15.9
a_arr = np.linspace(0.1, 1.0, 10)
lk_arr = np.log(k_data)  # Using loaded k_data

# Generate power spectrum responses using pkresponse.py functions
generated_Pgm_resp = darkemu_Pgm_resp(cosmo, hmc, prof_hod, deltah=deltah, log10Mh_min=log10Mh_min, log10Mh_max=log10Mh_max, lk_arr=lk_arr, a_arr=a_arr)
generated_Pgg_resp = darkemu_Pgg_resp(cosmo, hmc, prof_hod, deltalnAs=0.03, log10Mh_min=log10Mh_min, log10Mh_max=log10Mh_max, lk_arr=lk_arr, a_arr=a_arr)

# Compare the generated responses with simulation data within 5% accuracy
pgm_close = np.allclose(Pgm_resp_data, generated_Pgm_resp, rtol=0.05)
pgg_close = np.allclose(Pgg_resp_data, generated_Pgg_resp, rtol=0.05)

print(f"Pgm response close to simulation data within 5% accuracy: {pgm_close}")
print(f"Pgg response close to simulation data within 5% accuracy: {pgg_close}")